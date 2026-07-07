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
