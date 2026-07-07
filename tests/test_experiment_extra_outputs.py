"""The engine merges extra_outputs() from ANY algorithm, not just LatentModule.

Isolates the experiment.py hook (the Cflows GRN head rides this path) from the
heavy ODE training + the MPS/dopri5 float64 issue, via a tiny fake LightningModule.
"""
import numpy as np
import pytest

pytest.importorskip("lightning")
import torch  # noqa: E402
from lightning import LightningModule, Trainer  # noqa: E402

from manylatents.data.precomputed_datamodule import PrecomputedDataModule  # noqa: E402
from manylatents.experiment import run_experiment  # noqa: E402


class _FakeLit(LightningModule):
    """Minimal LightningModule with an encode() + extra_outputs() (like a GRN head)."""

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def training_step(self, batch, batch_idx):
        return (self.p ** 2).sum()

    def test_step(self, batch, batch_idx):
        self.log("test_loss", (self.p ** 2).sum())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def encode(self, x):
        return torch.as_tensor(x)[:, :2]

    def extra_outputs(self):
        return {"grn_edges": np.array([[0, 1], [1, 2]]), "grn_weights": np.array([0.5, -1.0])}


def test_run_experiment_merges_lightning_extra_outputs():
    X = np.random.default_rng(0).random((12, 4)).astype(np.float32)
    dm = PrecomputedDataModule(data=X, batch_size=12)
    trainer = Trainer(
        accelerator="cpu", devices=1, max_epochs=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
    )
    results = run_experiment(datamodule=dm, algorithm=_FakeLit(), trainer=trainer)
    assert "embeddings" in results
    # the hook: a non-LatentModule algorithm's extra_outputs are merged
    assert "grn_edges" in results and "grn_weights" in results
    assert np.asarray(results["grn_edges"]).shape == (2, 2)
