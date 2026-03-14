"""Tests for Archetypal synthetic dataset and ArchetypalDataModule."""
import numpy as np
import pytest

from manylatents.data.synthetic_dataset import Archetypal, SyntheticDataset
from manylatents.data.archetypal import ArchetypalDataModule


# ---------------------------------------------------------------------------
# Archetypal dataset
# ---------------------------------------------------------------------------

def test_isinstance_synthetic_dataset():
    ds = Archetypal(n_components=3, n_obs=100, random_state=0)
    assert isinstance(ds, SyntheticDataset)


def test_shape_flat_simplex():
    # project_to_sphere=False → ambient dim == n_components
    ds = Archetypal(n_components=3, n_obs=200, random_state=0, project_to_sphere=False)
    assert ds.data.shape == (200, 3)


def test_shape_sphere_projection():
    # project_to_sphere=True → ambient dim == n_components + 1
    ds = Archetypal(n_components=3, n_obs=200, random_state=0, project_to_sphere=True)
    assert ds.data.shape == (200, 4)


def test_shape_output_dims():
    ds = Archetypal(n_components=3, n_obs=100, random_state=0,
                    project_to_sphere=False, output_dims=50)
    assert ds.data.shape == (100, 50)


def test_labels_shape_and_range():
    n_components = 4
    ds = Archetypal(n_components=n_components, n_obs=500, random_state=0)
    assert ds.metadata.shape == (500,)
    assert ds.metadata.min() >= 0
    assert ds.metadata.max() <= n_components


def test_determinism():
    ds1 = Archetypal(n_components=3, n_obs=100, random_state=7)
    ds2 = Archetypal(n_components=3, n_obs=100, random_state=7)
    np.testing.assert_array_equal(ds1.data, ds2.data)
    np.testing.assert_array_equal(ds1.metadata, ds2.metadata)


def test_different_seeds_differ():
    ds1 = Archetypal(n_components=3, n_obs=100, random_state=1)
    ds2 = Archetypal(n_components=3, n_obs=100, random_state=2)
    assert not np.array_equal(ds1.data, ds2.data)


def test_len():
    ds = Archetypal(n_components=3, n_obs=123, random_state=0)
    assert len(ds) == 123


def test_getitem_keys():
    ds = Archetypal(n_components=3, n_obs=50, random_state=0)
    item = ds[0]
    assert "data" in item and "metadata" in item


def test_use_gap_reduces_obs():
    ds = Archetypal(n_components=4, n_obs=500, random_state=0,
                    use_gap=True, n_gaps=1)
    # At least one vertex cluster removed — dataset should be smaller
    assert len(ds) < 500


def test_save_figure_default_is_false(tmp_path):
    # Constructing with default save_figure=False must not write any files
    Archetypal(n_components=3, n_obs=100, random_state=0, save_dir=str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_save_figure_opt_in(tmp_path):
    Archetypal(n_components=3, n_obs=100, random_state=0,
               save_figure=True, save_dir=str(tmp_path))
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) > 0


# ---------------------------------------------------------------------------
# ArchetypalDataModule
# ---------------------------------------------------------------------------

def test_datamodule_setup_full():
    dm = ArchetypalDataModule(n_components=3, n_obs=200, random_state=0, mode="full")
    dm.setup()
    assert dm.train_dataset is not None
    assert dm.test_dataset is not None


def test_datamodule_setup_split():
    dm = ArchetypalDataModule(n_components=3, n_obs=200, random_state=0, mode="split")
    dm.setup()
    assert len(dm.train_dataset) + len(dm.test_dataset) == 200


def test_datamodule_train_dataloader():
    dm = ArchetypalDataModule(n_components=3, n_obs=100, batch_size=32,
                              random_state=0, mode="full")
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert "data" in batch


def test_datamodule_save_figure_default_is_false(tmp_path):
    dm = ArchetypalDataModule(n_components=3, n_obs=100, random_state=0,
                              save_dir=str(tmp_path))
    dm.setup()
    assert list(tmp_path.iterdir()) == []
