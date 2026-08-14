"""Per-cell `time` plumbing through api.run → datamodule → batch (for trajectory models)."""
import numpy as np
import pytest


def _xy(n=20, d=5, k=4):
    X = np.random.default_rng(0).random((n, d)).astype(np.float32)
    t = np.repeat(np.arange(k), n // k).astype(float)
    return X, t


def test_datamodule_puts_time_in_the_batch():
    from manylatents.data.precomputed_datamodule import PrecomputedDataModule

    X, t = _xy()
    dm = PrecomputedDataModule(data=X, time=t, batch_size=8)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert "time" in batch and batch["time"].shape == (8,)
    assert dm.train_dataset.get_time().shape == (20,)


def test_time_absent_when_not_provided():
    from manylatents.data.precomputed_datamodule import PrecomputedDataModule

    X, _ = _xy()
    dm = PrecomputedDataModule(data=X, batch_size=8)
    dm.setup()
    assert "time" not in next(iter(dm.train_dataloader()))
    assert dm.train_dataset.get_time() is None


def test_api_run_time_is_a_backward_compatible_noop():
    from manylatents import api

    X, t = _xy()
    base = api.run(input_data=X, algorithm="pca", seed=0)
    witht = api.run(input_data=X, algorithm="pca", seed=0, time=t)
    # pca ignores time; passing it must not error and must not change the result
    assert np.allclose(np.asarray(base["embeddings"]), np.asarray(witht["embeddings"]))


def test_latent_ode_ignores_time():
    """`api.run`'s docstring used to promise LatentODE reads batch['time']. It does not.

    Its ``t_span`` is the ``integration_times`` *hyperparameter*
    (``latent_ode.py:100``), so the timepoints cannot reach it. Pinned here because the
    claim is otherwise invisible: nothing errors, the run just ignores the argument.
    If LatentODE ever does grow a batch-time seam this test fails and the docstring
    (api.py, precomputed_dataset.py) must be rewritten with it.
    """
    pytest.importorskip("torchdiffeq")
    import torch
    from lightning import Trainer

    from manylatents.api import _instantiate_lightning, _lightning_config
    from manylatents.data.precomputed_datamodule import PrecomputedDataModule
    from manylatents.experiment import run_experiment

    X, t = _xy()

    def embed(time):
        torch.manual_seed(0)
        dm = PrecomputedDataModule(data=X, time=time, batch_size=len(X))
        mod = _instantiate_lightning(_lightning_config("latent_ode"), dm)
        # accelerator="cpu": "auto" picks MPS on macOS and torchdiffeq's float64 solver
        # tolerances crash there (pre-existing, unrelated to the time channel).
        trainer = Trainer(
            accelerator="cpu", devices=1, max_epochs=2, logger=False,
            enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, num_sanity_val_steps=0,
        )
        out = run_experiment(datamodule=dm, algorithm=mod, trainer=trainer, seed=0)
        return np.asarray(out["embeddings"])

    # Same seed, three different time channels — byte-identical, not merely close.
    assert np.array_equal(embed(t), embed(t * 100.0))
    assert np.array_equal(embed(t), embed(None))
