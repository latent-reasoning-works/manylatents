"""The engine collects outputs off ANY algorithm, not just LatentModule.

Isolates the experiment.py hook (the Cflows GRN head and MIOFlow's trajectories both
ride this path) from the heavy ODE training + the MPS/dopri5 float64 issue, via tiny
fake LightningModules. Both halves of the merge are covered: an algorithm-specific
``extra_outputs()`` and a generic registry output with no ``extra_outputs()`` at all.
"""
import numpy as np
import pytest

pytest.importorskip("lightning")
import torch  # noqa: E402
from lightning import LightningModule, Trainer  # noqa: E402

from manylatents.data.precomputed_datamodule import PrecomputedDataModule  # noqa: E402
from manylatents.experiment import run_experiment  # noqa: E402


class _TrainableLit(LightningModule):
    """Just enough LightningModule to survive one epoch and produce an embedding."""

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


class _FakeLit(_TrainableLit):
    """Algorithm-SPECIFIC outputs via extra_outputs(), like Cflows' GRN head."""

    def extra_outputs(self):
        return {"grn_edges": np.array([[0, 1], [1, 2]]), "grn_weights": np.array([0.5, -1.0])}


class _FakeTrajLit(_TrainableLit):
    """MIOFlow's shape: stores ``.trajectories``, defines NO ``extra_outputs()``.

    Before the registry the generic collect was an inherited LatentModule method, so a
    module of this shape could not emit trajectories however it was written (#295) —
    MIOFlow computes them at ``on_train_end`` and nothing ever read them.
    """

    def __init__(self):
        super().__init__()
        self.trajectories = torch.randn(5, 12, 2)


def _run(algorithm):
    X = np.random.default_rng(0).random((12, 4)).astype(np.float32)
    dm = PrecomputedDataModule(data=X, batch_size=12)
    trainer = Trainer(
        accelerator="cpu", devices=1, max_epochs=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
    )
    return run_experiment(datamodule=dm, algorithm=algorithm, trainer=trainer)


def test_run_experiment_merges_lightning_extra_outputs():
    results = _run(_FakeLit())
    assert "embeddings" in results
    # the hook: a non-LatentModule algorithm's extra_outputs are merged
    assert "grn_edges" in results and "grn_weights" in results
    assert np.asarray(results["grn_edges"]).shape == (2, 2)


def test_run_experiment_collects_generic_outputs_off_a_lightning_module():
    """#295, framework-free: no LatentModule anywhere, no extra_outputs() at all, and
    the trajectories still land in the results dict as numpy."""
    results = _run(_FakeTrajLit())
    assert isinstance(results["trajectories"], np.ndarray)
    assert results["trajectories"].shape == (5, 12, 2)
