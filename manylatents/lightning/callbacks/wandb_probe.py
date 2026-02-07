# manylatents/lightning/callbacks/wandb_probe.py
"""WandB logging for representation probing."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class WandbProbeLogger:
    """Log representation probe results to WandB.

    Logs:
    - Trajectory spread over time
    - Operator statistics (spectral properties)
    - Optional: Operator heatmaps as images

    Attributes:
        log_spread: Log spread metrics
        log_operator_stats: Log operator statistics
        log_images: Log operator heatmaps as images
        prefix: Prefix for all logged keys
    """
    log_spread: bool = True
    log_operator_stats: bool = True
    log_images: bool = False
    prefix: str = "probe"

    def log_trajectory(
        self,
        trajectory: List[Tuple[int, np.ndarray]],
        step: Optional[int] = None,
    ):
        """Log trajectory statistics to wandb.

        Args:
            trajectory: List of (step, operator) tuples
            step: Global step for logging (uses last trajectory step if None)
        """
        if not WANDB_AVAILABLE or wandb.run is None:
            return

        if not trajectory:
            return

        log_step = step if step is not None else trajectory[-1][0]
        metrics = {}

        if self.log_spread and len(trajectory) > 1:
            spread = self._compute_spread(trajectory)
            metrics[f"{self.prefix}/spread"] = spread

        if self.log_operator_stats:
            # Stats for latest operator
            _, latest_op = trajectory[-1]
            stats = self._compute_operator_stats(latest_op)
            for k, v in stats.items():
                metrics[f"{self.prefix}/{k}"] = v

        if self.log_images and len(trajectory) > 0:
            _, latest_op = trajectory[-1]
            metrics[f"{self.prefix}/operator"] = wandb.Image(
                latest_op,
                caption=f"Diffusion operator at step {log_step}",
            )

        wandb.log(metrics, step=log_step)

    def log_multi_model_spread(
        self,
        trajectories: List[List[Tuple[int, np.ndarray]]],
        model_names: Optional[List[str]] = None,
        step: Optional[int] = None,
    ):
        """Log spread across multiple models at current step.

        Args:
            trajectories: List of trajectories, one per model
            model_names: Optional names for models
            step: Global step for logging
        """
        if not WANDB_AVAILABLE or wandb.run is None:
            return

        from manylatents.callbacks.probing import compute_multi_model_spread

        spreads = compute_multi_model_spread(trajectories)

        metrics = {}
        for i, spread in enumerate(spreads):
            metrics[f"{self.prefix}/multi_model_spread_step_{i}"] = spread

        # Also log latest spread
        if len(spreads) > 0:
            metrics[f"{self.prefix}/multi_model_spread_latest"] = spreads[-1]

        wandb.log(metrics, step=step)

    def _compute_spread(
        self,
        trajectory: List[Tuple[int, np.ndarray]]
    ) -> float:
        """Compute spread of operators in trajectory."""
        if len(trajectory) < 2:
            return 0.0

        ops = [op for _, op in trajectory]
        dists = []
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                dists.append(np.linalg.norm(ops[i] - ops[j], "fro"))
        return float(np.mean(dists))

    def _compute_operator_stats(self, op: np.ndarray) -> Dict[str, float]:
        """Compute statistics of a diffusion operator."""
        stats = {}

        # Spectral properties
        eigvals = np.linalg.eigvalsh(op)
        sorted_eigvals = np.sort(np.abs(eigvals))[::-1]
        stats["spectral_gap"] = float(1.0 - sorted_eigvals[1]) if len(sorted_eigvals) > 1 else 0.0
        stats["spectral_radius"] = float(np.max(np.abs(eigvals)))

        # Entropy of stationary distribution (if row-stochastic)
        row_sums = op.sum(axis=1)
        if np.allclose(row_sums, 1.0, rtol=1e-3):
            # Approximate stationary distribution via power iteration
            pi = np.ones(len(op)) / len(op)
            for _ in range(100):
                pi = pi @ op
            pi = pi / pi.sum()
            entropy = -np.sum(pi * np.log(pi + 1e-10))
            stats["stationary_entropy"] = float(entropy)

        return stats
