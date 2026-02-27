"""
LoadingsAnalysisCallback - Analyze shared vs modality-specific components in loadings.

For multi-modal fusion (e.g., DNA + RNA + Protein), this callback analyzes
which components capture shared structure vs modality-specific signal.

Hypothesis: "Shared" components (high loadings from multiple modalities)
represent evolutionarily conserved relationships.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from manylatents.callbacks.embedding.base import EmbeddingCallback, LatentOutputs

logger = logging.getLogger(__name__)


class LoadingsAnalysisCallback(EmbeddingCallback):
    """
    Analyze loadings to identify shared vs modality-specific components.

    For MergingModule with concat_pca or similar, this callback:
    1. Extracts loadings from module.get_loadings()
    2. Decomposes contributions by modality
    3. Classifies components as "shared" or "modality-specific"
    4. Logs analysis to wandb if available

    Example config:
        callbacks:
          embedding:
            loadings:
              _target_: manylatents.callbacks.embedding.loadings_analysis.LoadingsAnalysisCallback
              modality_dims: [1920, 256, 1536]  # DNA, RNA, Protein
              modality_names: [dna, rna, protein]
              threshold: 0.1
    """

    def __init__(
        self,
        modality_dims: Optional[List[int]] = None,
        modality_names: Optional[List[str]] = None,
        threshold: float = 0.1,
        log_to_wandb: bool = True,
    ) -> None:
        """
        Initialize LoadingsAnalysisCallback.

        Args:
            modality_dims: List of dimensions for each modality in concatenated order.
                If None, will try to infer from module.get_loadings().channel_ranges.
            modality_names: Names for each modality. If None, uses ["modality_0", ...].
            threshold: Minimum relative contribution to count as "contributing".
                A component is "shared" if all modalities contribute above threshold.
            log_to_wandb: Whether to log results to wandb.
        """
        super().__init__()
        self.modality_dims = modality_dims
        self.modality_names = modality_names
        self.threshold = threshold
        self.log_to_wandb = log_to_wandb

        logger.info(
            f"LoadingsAnalysisCallback initialized: "
            f"threshold={threshold}, log_to_wandb={log_to_wandb}"
        )

    def _compute_modality_contributions(
        self,
        loadings: np.ndarray,
        modality_ranges: Dict[str, tuple],
    ) -> Dict[str, np.ndarray]:
        """Compute per-modality contribution to each component.

        Args:
            loadings: Full loadings matrix (n_features, n_components).
            modality_ranges: Dict mapping modality name to (start, end) indices.

        Returns:
            Dict mapping modality name to contribution array (n_components,).
            Each value is the L2 norm of that modality's loadings for each component.
        """
        contributions = {}
        for name, (start, end) in modality_ranges.items():
            modality_loadings = loadings[start:end, :]
            # L2 norm per component
            contributions[name] = np.linalg.norm(modality_loadings, axis=0)
        return contributions

    def _classify_components(
        self,
        contributions: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Classify components as shared or modality-specific.

        Args:
            contributions: Dict of per-modality contribution arrays.

        Returns:
            Dict with classification results.
        """
        n_components = next(iter(contributions.values())).shape[0]
        modalities = list(contributions.keys())

        # Normalize contributions per component (sum to 1)
        total_per_component = np.zeros(n_components)
        for contrib in contributions.values():
            total_per_component += contrib
        total_per_component = np.maximum(total_per_component, 1e-10)

        normalized = {
            name: contrib / total_per_component
            for name, contrib in contributions.items()
        }

        # Classify each component
        shared_mask = np.ones(n_components, dtype=bool)
        dominant_modality = []

        for i in range(n_components):
            component_contribs = {name: normalized[name][i] for name in modalities}

            # Check if all modalities contribute above threshold
            all_above = all(c >= self.threshold for c in component_contribs.values())
            shared_mask[i] = all_above

            # Find dominant modality
            dominant = max(component_contribs, key=component_contribs.get)
            dominant_modality.append(dominant)

        n_shared = int(shared_mask.sum())

        return {
            "n_shared": n_shared,
            "n_specific": n_components - n_shared,
            "shared_fraction": n_shared / n_components if n_components > 0 else 0,
            "shared_mask": shared_mask,
            "dominant_modality": dominant_modality,
            "normalized_contributions": normalized,
        }

    def on_latent_end(
        self,
        dataset: Any,
        embeddings: LatentOutputs,
        module: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Analyze loadings from module.

        Args:
            dataset: Dataset object (unused).
            embeddings: LatentOutputs dict.
            module: LatentModule with get_loadings() method.

        Returns:
            Dict with analysis results.
        """
        if module is None:
            logger.warning("No module provided - cannot analyze loadings")
            return {}

        if not hasattr(module, "get_loadings"):
            logger.info(
                f"Module {type(module).__name__} has no get_loadings() - skipping analysis"
            )
            return {}

        try:
            loadings_obj = module.get_loadings()
        except RuntimeError as e:
            logger.warning(f"Could not get loadings: {e}")
            return {}

        # Get loadings matrix and channel ranges
        if hasattr(loadings_obj, "components") and loadings_obj.components is not None:
            loadings = loadings_obj.components
            modality_ranges = loadings_obj.channel_ranges
        else:
            logger.warning("Loadings object has no components matrix")
            return {}

        # Override with provided modality info if available
        if self.modality_dims is not None:
            # Build ranges from dims
            modality_ranges = {}
            offset = 0
            names = self.modality_names or [f"modality_{i}" for i in range(len(self.modality_dims))]
            for i, dim in enumerate(self.modality_dims):
                name = names[i] if i < len(names) else f"modality_{i}"
                modality_ranges[name] = (offset, offset + dim)
                offset += dim

        logger.info(f"Analyzing loadings: shape={loadings.shape}, modalities={list(modality_ranges.keys())}")

        # Compute contributions
        contributions = self._compute_modality_contributions(loadings, modality_ranges)

        # Classify components
        classification = self._classify_components(contributions)

        # Build results
        results = {
            "n_shared": classification["n_shared"],
            "n_specific": classification["n_specific"],
            "shared_fraction": classification["shared_fraction"],
        }

        # Add per-modality contribution summary
        for name, contrib in contributions.items():
            results[f"{name}_mean_contribution"] = float(np.mean(contrib))
            results[f"{name}_total_contribution"] = float(np.sum(contrib))

        # Compute explained variance in shared components if available
        if hasattr(loadings_obj, "explained_variance_ratio") and loadings_obj.explained_variance_ratio is not None:
            evr = loadings_obj.explained_variance_ratio
            shared_variance = float(evr[classification["shared_mask"]].sum())
            results["shared_variance_ratio"] = shared_variance
            logger.info(f"Shared components explain {shared_variance:.2%} of variance")

        logger.info(
            f"Loadings analysis: {results['n_shared']} shared, "
            f"{results['n_specific']} specific components"
        )

        # Log to wandb if available and enabled
        if self.log_to_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({f"loadings/{k}": v for k, v in results.items()})
                    logger.info("Logged loadings analysis to wandb")
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")

        # Register outputs
        for key, value in results.items():
            self.register_output(key, value)

        return results
