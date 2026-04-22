"""Tests for TuningFork synthetic dataset and TuningForkDataModule."""
import numpy as np
import pytest
from manylatents.data.synthetic_dataset import TuningFork, SyntheticDataset


def _make(n_prong=50, **kw):
    return TuningFork(n_prong=n_prong, random_state=0, **kw)


# ---------------------------------------------------------------------------
# Geometry and sampling
# ---------------------------------------------------------------------------

def test_isinstance_synthetic_dataset():
    assert isinstance(_make(), SyntheticDataset)


def test_shape_default():
    ds = _make(n_prong=50, handle_prong_ratio=0.2)
    n_handle = int(0.2 * 50)
    assert ds.data.shape == (n_handle + 2 * 50, 2)


def test_shape_total_n():
    ds = _make(n_prong=100, handle_prong_ratio=0.5)
    n_handle = int(0.5 * 100)
    assert ds.data.shape == (n_handle + 200, 2)


def test_labels_three_values():
    ds = _make(n_prong=60, handle_prong_ratio=0.3)
    assert set(ds.metadata.tolist()) == {0, 1, 2}


def test_label_counts():
    n_prong = 80
    ratio = 0.25
    ds = TuningFork(n_prong=n_prong, handle_prong_ratio=ratio, random_state=0)
    n_handle = int(ratio * n_prong)
    assert np.sum(ds.metadata == 0) == n_handle
    assert np.sum(ds.metadata == 1) == n_prong
    assert np.sum(ds.metadata == 2) == n_prong


def test_determinism():
    ds1 = TuningFork(n_prong=50, random_state=7)
    ds2 = TuningFork(n_prong=50, random_state=7)
    np.testing.assert_array_equal(ds1.data, ds2.data)
    np.testing.assert_array_equal(ds1.metadata, ds2.metadata)


def test_different_seeds_differ():
    ds1 = TuningFork(n_prong=50, random_state=1)
    ds2 = TuningFork(n_prong=50, random_state=2)
    assert not np.array_equal(ds1.data, ds2.data)


def test_len():
    ds = _make(n_prong=50, handle_prong_ratio=0.4)
    n_handle = int(0.4 * 50)
    assert len(ds) == n_handle + 100


def test_getitem_keys():
    ds = _make()
    item = ds[0]
    assert "data" in item and "metadata" in item


def test_prongs_close_in_euclidean():
    ds = TuningFork(n_prong=200, dist_between_prongs=0.1, prong_length=3.0,
                    noise=0.0, random_state=0)
    left = ds.data[ds.metadata == 1]
    right = ds.data[ds.metadata == 2]
    from sklearn.metrics import pairwise_distances
    dists = pairwise_distances(left[:10], right[:10])
    assert dists.min() < 0.5


def test_rotate_to_dim():
    n_prong = 50
    ratio = 0.2
    ds = TuningFork(n_prong=n_prong, handle_prong_ratio=ratio,
                    rotate_to_dim=50, random_state=0)
    n_handle = int(ratio * n_prong)
    assert ds.data.shape == (n_handle + 2 * n_prong, 50)


def test_handle_sparser_than_prongs():
    ds = TuningFork(n_prong=100, handle_prong_ratio=0.2, prong_length=3.0,
                    handle_length=2.0, noise=0.0, random_state=0)
    n_handle = np.sum(ds.metadata == 0)
    n_prong = np.sum(ds.metadata == 1)
    assert n_handle < n_prong


# ---------------------------------------------------------------------------
# Ground truth distances
# ---------------------------------------------------------------------------

def test_gt_dists_shape():
    ds = _make(n_prong=40, handle_prong_ratio=0.25)
    n = len(ds)
    D = ds.get_gt_dists()
    assert D.shape == (n, n)


def test_gt_dists_symmetric():
    ds = _make(n_prong=40)
    D = ds.get_gt_dists()
    np.testing.assert_allclose(D, D.T, atol=1e-10)


def test_gt_dists_diagonal_zero():
    ds = _make(n_prong=40)
    D = ds.get_gt_dists()
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)


def test_gt_dists_cross_prong_larger_than_same_prong():
    ds = TuningFork(n_prong=200, dist_between_prongs=0.1, noise=0.0, random_state=0)
    D = ds.get_gt_dists()
    left_idx = np.where(ds.metadata == 1)[0]
    right_idx = np.where(ds.metadata == 2)[0]
    same_prong_dists = D[left_idx[:5], :][:, left_idx[5:10]].flatten()
    cross_prong_dists = D[left_idx[:5], :][:, right_idx[:5]].flatten()
    assert cross_prong_dists.mean() > same_prong_dists.mean()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def test_save_viz_default_false(tmp_path):
    TuningFork(n_prong=30, random_state=0, save_dir=str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_save_viz_creates_png(tmp_path):
    TuningFork(n_prong=30, random_state=0, save_viz=True, save_dir=str(tmp_path))
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 1


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

from manylatents.data.tuning_fork import TuningForkDataModule


def test_datamodule_setup_full():
    dm = TuningForkDataModule(n_prong=50, mode="full", random_state=0)
    dm.setup()
    assert dm.train_dataset is not None
    assert dm.test_dataset is not None


def test_datamodule_setup_split():
    dm = TuningForkDataModule(n_prong=100, handle_prong_ratio=0.2,
                              mode="split", test_split=0.2, random_state=0)
    dm.setup()
    n_handle = int(0.2 * 100)
    total = n_handle + 200
    assert len(dm.train_dataset) + len(dm.test_dataset) == total


def test_datamodule_train_dataloader():
    dm = TuningForkDataModule(n_prong=50, batch_size=16, mode="full", random_state=0)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert "data" in batch and "metadata" in batch


def test_datamodule_save_viz_default_false(tmp_path):
    dm = TuningForkDataModule(n_prong=30, random_state=0, save_dir=str(tmp_path))
    dm.setup()
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Hydra config smoke tests
# ---------------------------------------------------------------------------

from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import hydra.utils
import manylatents.configs  # noqa: F401 — registers ConfigStore

CONFIGS = Path(__file__).parent.parent / "manylatents" / "configs"


def _compose(overrides):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIGS.resolve()), version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


def test_hydra_instantiate_positive():
    cfg = _compose(["data=tuning_fork_positive"])
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup()
    assert dm.train_dataset is not None


def test_hydra_instantiate_dense_control():
    cfg = _compose(["data=tuning_fork_dense_control"])
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup()
    assert dm.train_dataset is not None


def test_hydra_instantiate_sparse_control():
    cfg = _compose(["data=tuning_fork_sparse_control"])
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup()
    assert dm.train_dataset is not None
