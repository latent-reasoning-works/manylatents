"""SaveTrajectory callback — writes per-step embedding snapshots to disk.

Also provides ``load_trajectory()`` for reading back what was written,
returning ``list[EmbeddingOutputs]`` (one dict per step).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from manylatents.callbacks.embedding.base import EmbeddingCallback, EmbeddingOutputs

logger = logging.getLogger(__name__)


class SaveTrajectory(EmbeddingCallback):
    """Write trajectory manifest + per-step NPY files from pipeline snapshots.

    Reads ``step_snapshots`` from the embeddings dict (populated by
    ``run_pipeline``) and writes:

    - ``trajectory.json`` — manifest with step metadata and file references
    - ``step_{i}_embedding.npy`` — embedding array for each step

    If ``step_snapshots`` is absent (single-algorithm run), falls back to
    saving the final embedding as a single-step trajectory.
    """

    def __init__(
        self,
        save_dir: str = "outputs",
        experiment_name: str = "experiment",
    ) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        os.makedirs(self.save_dir, exist_ok=True)

    def on_latent_end(self, dataset, embeddings: dict) -> dict:
        snapshots = embeddings.get("step_snapshots")

        if snapshots:
            steps_meta = self._write_snapshots(snapshots)
        else:
            # Single-algorithm run: synthesize a one-step trajectory
            final_emb = embeddings["embeddings"]
            if not isinstance(final_emb, np.ndarray):
                final_emb = np.asarray(final_emb)
            npy_name = "step_0_embedding.npy"
            np.save(os.path.join(self.save_dir, npy_name), final_emb)

            algo = (embeddings.get("metadata", {})
                    .get("final_algorithm_type", "unknown"))
            steps_meta = [{
                "step_index": 0,
                "step_name": "single_step",
                "algorithm": algo,
                "output_shape": list(final_emb.shape),
                "embedding_file": npy_name,
                "step_time": None,
            }]

        # Save label array if present
        label = embeddings.get("label")
        label_file = None
        if label is not None:
            label_arr = label if isinstance(label, np.ndarray) else np.asarray(label)
            label_file = "label.npy"
            np.save(os.path.join(self.save_dir, label_file), label_arr)

        final_emb = embeddings["embeddings"]
        if not isinstance(final_emb, np.ndarray):
            final_emb = np.asarray(final_emb)

        scores = embeddings.get("scores", {})
        # Flatten (scalar, array) tuples to just the scalar for the manifest
        flat_scores = {}
        for k, v in scores.items():
            if isinstance(v, tuple) and len(v) == 2:
                flat_scores[k] = float(v[0])
            elif isinstance(v, (int, float)):
                flat_scores[k] = float(v)
            else:
                try:
                    flat_scores[k] = float(v)
                except (TypeError, ValueError):
                    pass  # skip non-scalar scores in manifest

        manifest = {
            "version": "1.0",
            "experiment_name": self.experiment_name,
            "steps": steps_meta,
            "final_shape": list(final_emb.shape),
            "scores": flat_scores,
            "label_file": label_file,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        manifest_path = os.path.join(self.save_dir, "trajectory.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            f"SaveTrajectory: wrote {len(steps_meta)} step(s) to {self.save_dir}"
        )

        self.register_output("trajectory_manifest", manifest_path)
        return self.callback_outputs

    def _write_snapshots(self, snapshots: list) -> list:
        """Write per-step NPY files and return manifest step entries."""
        steps_meta = []
        for snap in snapshots:
            idx = snap["step_index"]
            npy_name = f"step_{idx}_embedding.npy"
            np.save(os.path.join(self.save_dir, npy_name), snap["embedding"])
            steps_meta.append({
                "step_index": idx,
                "step_name": snap["step_name"],
                "algorithm": snap["algorithm"],
                "output_shape": snap["output_shape"],
                "embedding_file": npy_name,
                "step_time": snap.get("step_time"),
            })
        return steps_meta


def load_trajectory(path: str | Path) -> list[EmbeddingOutputs]:
    """Read a trajectory directory written by :class:`SaveTrajectory`.

    Returns one :data:`EmbeddingOutputs` dict per pipeline step, ordered by
    step index.  Scores are attached to the **last** step only (that is where
    ``run_pipeline`` evaluates metrics).

    Parameters
    ----------
    path : str | Path
        Directory containing ``trajectory.json`` and the referenced NPY files.

    Returns
    -------
    list[EmbeddingOutputs]
        One dict per step with keys: ``embeddings``, ``label`` (or ``None``),
        ``metadata``, ``scores``.
    """
    path = Path(path)
    with open(path / "trajectory.json") as f:
        manifest = json.load(f)

    # Load label once (shared across all steps)
    label_file = manifest.get("label_file")
    label = np.load(path / label_file) if label_file else None

    steps = manifest["steps"]
    scores = manifest.get("scores", {})
    result: list[EmbeddingOutputs] = []

    for i, step in enumerate(steps):
        is_last = i == len(steps) - 1
        result.append({
            "embeddings": np.load(path / step["embedding_file"]),
            "label": label,
            "metadata": {
                "step_index": step["step_index"],
                "step_name": step["step_name"],
                "algorithm": step["algorithm"],
                "step_time": step.get("step_time"),
            },
            "scores": scores if is_last else {},
        })

    return result
