"""Tests for Archetypal synthetic dataset and ArchetypalDataModule."""
import numpy as np
import pytest
from scipy.spatial.distance import pdist

from manylatents.data.synthetic_dataset import Archetypal, SyntheticDataset
from manylatents.data.archetypal import ArchetypalDataModule


# ---------------------------------------------------------------------------
# Archetypal dataset
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('n_vertices', range(2, 9))
@pytest.mark.parametrize('simplex_radius', [0.25, 1.0, 3.5])
def test_regular_simplex_vertices(n_vertices, simplex_radius):
    ds = Archetypal(n_components=n_vertices - 1, n_obs=10,
                    simplex_radius=simplex_radius, project_to_sphere=False)
    assert ds.vertices.shape == (n_vertices, n_vertices - 1)
    np.testing.assert_allclose(ds.vertices.mean(axis=0), 0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(pdist(ds.vertices), simplex_radius, atol=1e-12, rtol=0)


@pytest.mark.parametrize('expected', [
    # Helmert rows divided by sqrt(2), giving unit edge length.
    [[0.5, 0.2886751345948129],
     [-0.5, 0.2886751345948129],
     [0.0, -0.5773502691896258]],
    [[0.5, 0.2886751345948129, 0.2041241452319315],
     [-0.5, 0.2886751345948129, 0.2041241452319315],
     [0.0, -0.5773502691896258, 0.2041241452319315],
     [0.0, 0.0, -0.6123724356957945]],
])
def test_helmert_vertex_coordinates(expected):
    ds = Archetypal(n_components=len(expected) - 1, n_obs=10,
                    project_to_sphere=False)
    np.testing.assert_allclose(ds.vertices, expected, atol=1e-12, rtol=0)


def test_default_determinism_and_original_random_stream(monkeypatch):
    first = Archetypal()
    first_state = np.random.get_state()
    first_draw = np.random.random(10)
    second = Archetypal()
    np.testing.assert_array_equal(first.data, second.data)
    np.testing.assert_array_equal(first.gt_points, second.gt_points)
    np.testing.assert_array_equal(first.metadata, second.metadata)
    np.testing.assert_array_equal(first_draw, np.random.random(10))

    def original_conversion(self, X_bary, simplex_radius):
        n_vertices = X_bary.shape[1]
        centered = np.eye(n_vertices) - 1.0 / n_vertices
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        vertices = u[:, :n_vertices - 1] * s[:n_vertices - 1]
        edge_len = np.linalg.norm(vertices[0] - vertices[1])
        if edge_len > 0:
            vertices = vertices / edge_len * simplex_radius
        return X_bary @ vertices

    monkeypatch.setattr(Archetypal, '_barycentric_to_euclidean', original_conversion)
    original = Archetypal()
    original_state = np.random.get_state()
    for got, want in zip(first_state, original_state):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(first_draw, np.random.random(10))
    np.testing.assert_array_equal(first.metadata, original.metadata)


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


# Captured on darwin-arm64 after the closed-form simplex. Linux CI checks the same
# numbers, so Archetypal data for one seed is the same on every platform.
PORTABLE_ROWS = [
    [-0.4012172586640314, 0.20418588986870154, -0.185298745207738,
     -0.9241738400761759, 0.22154880700227125, -0.3600348286242663],
    [-0.37391367900583194, 0.13468504045527574, 0.0020483155015648957,
     -0.7954375868874224, 0.04085364001739655, -0.3729769946110165],
    [-0.3787211046936304, -0.4429219844168812, 0.14312737917240081,
     -0.5665537343338398, -0.5254036948777585, 0.05107944512612311],
]


def test_same_data_on_every_platform():
    from manylatents.data.synthetic_dataset import Archetypal

    ds = Archetypal(n_obs=30, noise=0.2, output_dims=6, random_state=7)
    assert ds.data.shape == (30, 6)
    assert ds.metadata[:3].tolist() == [3, 3, 1]
    np.testing.assert_allclose(ds.data[:3], PORTABLE_ROWS, rtol=1e-9, atol=1e-12)
