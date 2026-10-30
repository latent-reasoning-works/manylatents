"""Tests for the ReebGraphModule cell cap (``max_N``)."""
import inspect
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ripser = pytest.importorskip("ripser")
gudhi = pytest.importorskip("gudhi")

from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "manylatents" / "configs" / "algorithms" / "latent"
REEB_CONFIGS = [
    "reeb_graph.yaml",
    "reeb_graph_density.yaml",
    "reeb_graph_diffusion1.yaml",
    "reeb_graph_pca1.yaml",
]


def _make_swissroll(n=200, seed=42):
    """Generate a simple Swiss roll dataset (copied from tests/test_reeb_graph.py)."""
    rng = np.random.RandomState(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.rand(n))
    x = t * np.cos(t)
    y = 30 * rng.rand(n)
    z = t * np.sin(t)
    return np.column_stack([x, y, z])


@pytest.fixture(scope="module")
def x200():
    return torch.from_numpy(_make_swissroll(n=200, seed=42)).float()


def _np(t):
    return t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


# ------------------------------------------------------------------ constructor --


def test_defaults():
    m = ReebGraphModule()
    assert m.max_N == 5000
    assert m.random_state == 42
    assert m.subsample_indices is None


def test_invalid_cap():
    with pytest.raises(ValueError):
        ReebGraphModule(max_N=1)


def test_no_random_seed_parameter():
    params = inspect.signature(ReebGraphModule.__init__).parameters
    assert "max_N" in params
    assert "random_state" in params
    assert "random_seed" not in params


# ------------------------------------------------------------------- sampling --


def test_indices_match_persistent_homology_procedure(x200):
    m = ReebGraphModule(max_N=150, random_state=3, n_bins=5)
    m.fit(x200)
    got = m.extra_outputs()["subsample_indices"]
    np.testing.assert_array_equal(got, np.array(random.Random(3).sample(range(200), k=150)))
    state = random.getstate()
    try:
        random.seed(3)
        ph = np.array(random.sample(range(200), k=150))
    finally:
        random.setstate(state)
    np.testing.assert_array_equal(got, ph)


def test_global_random_state_untouched(x200):
    random.seed(123)
    a = random.random()
    random.seed(123)
    ReebGraphModule(max_N=150, n_bins=5).fit(x200)
    b = random.random()
    assert a == b


def test_pairwise_distances_on_sample_only(x200, monkeypatch):
    import manylatents.algorithms.latent.reeb_graph as rg

    real = rg.pairwise_distances
    shapes = []

    def recorder(X, *args, **kwargs):
        shapes.append(np.shape(X))
        return real(X, *args, **kwargs)

    monkeypatch.setattr(
        "manylatents.algorithms.latent.reeb_graph.pairwise_distances", recorder
    )
    ReebGraphModule(max_N=150, n_bins=5).fit(x200)
    assert shapes
    assert all(s == (150, 3) for s in shapes)


def test_no_cap_at_or_below_threshold(x200):
    fits = []
    for cap in (None, 200, 5000):
        m = ReebGraphModule(max_N=cap, n_bins=5)
        out = _np(m.fit_transform(x200))
        fits.append((out, m.adjacency(), m.node_coordinates))
        np.testing.assert_array_equal(m.subsample_indices, np.arange(200))
    for other in fits[1:]:
        for a, b in zip(fits[0], other):
            assert np.array_equal(a, b)


def test_seed_controls_sample(x200):
    ms = []
    for seed in (0, 1, 0):
        m = ReebGraphModule(max_N=150, n_bins=5, random_state=seed)
        m.fit(x200)
        ms.append(m)
    assert not np.array_equal(ms[0].subsample_indices, ms[1].subsample_indices)
    np.testing.assert_array_equal(ms[0].subsample_indices, ms[2].subsample_indices)
    np.testing.assert_array_equal(ms[0].adjacency(), ms[2].adjacency())
    np.testing.assert_array_equal(ms[0].node_coordinates, ms[2].node_coordinates)


def test_capped_graph_is_graph_of_sample(x200):
    m = ReebGraphModule(max_N=150, random_state=3, n_bins=5)
    m.fit(x200)
    m2 = ReebGraphModule(max_N=None, n_bins=5)
    m2.fit(x200[m.subsample_indices])
    np.testing.assert_array_equal(m.adjacency(), m2.adjacency())
    np.testing.assert_array_equal(m.node_coordinates, m2.node_coordinates)
    assert m.structural_summary == m2.structural_summary


# ----------------------------------------------------------------- membership --


def test_one_row_per_input_row(x200):
    m = ReebGraphModule(max_N=150, n_bins=5)
    out = m.fit_transform(x200)
    assert out.shape[0] == 200
    assert out.shape[1] == m.adjacency().shape[0]


def test_unsampled_row_copies_nearest(x200):
    m = ReebGraphModule(max_N=150, random_state=3, n_bins=5)
    out = _np(m.fit_transform(x200))
    x = _np(x200)
    idx = m.subsample_indices
    sampled = set(idx.tolist())
    for j in range(200):
        if j in sampled:
            continue
        d = np.linalg.norm(x[idx] - x[j], axis=1)
        k = idx[int(np.argmin(d))]
        np.testing.assert_array_equal(out[j], out[k])


def test_sampled_rows_keep_membership(x200):
    m = ReebGraphModule(max_N=150, random_state=3, n_bins=5)
    out = _np(m.fit_transform(x200))
    out2 = _np(
        ReebGraphModule(max_N=None, n_bins=5).fit_transform(x200[m.subsample_indices])
    )
    np.testing.assert_array_equal(out[m.subsample_indices], out2)


def test_duplicate_rows_keep_membership(x200):
    x = x200.clone()
    x[1] = x[0]
    m = ReebGraphModule(max_N=150, random_state=3, n_bins=5)
    out = _np(m.fit_transform(x))
    ref = _np(
        ReebGraphModule(max_N=None, n_bins=5).fit_transform(x[m.subsample_indices])
    )
    for pos, k in enumerate(m.subsample_indices):
        np.testing.assert_array_equal(out[k], ref[pos])


def test_unseen_rows_raise(x200):
    m = ReebGraphModule(max_N=150, n_bins=5)
    m.fit(x200)
    with pytest.raises(NotImplementedError):
        m.transform(x200[:50])


# ------------------------------------------------------------- extra_outputs --


@pytest.mark.parametrize("cap,n", [(150, 150), (None, 200)])
def test_extra_outputs_has_subsample_indices(x200, cap, n):
    m = ReebGraphModule(max_N=cap, n_bins=5)
    m.fit(x200)
    extras = m.extra_outputs()
    assert {"node_coordinates", "structural_summary", "subsample_indices"} <= set(extras)
    assert extras["subsample_indices"].dtype == np.int64
    assert extras["subsample_indices"].ndim == 1
    assert len(extras["subsample_indices"]) == n


def test_extra_outputs_empty_before_fit():
    assert ReebGraphModule().extra_outputs() == {}


def test_run_result_has_subsample_indices():
    from manylatents.api import run

    result = run(
        data="swissroll",
        algorithms={"latent": "reeb_graph"},
        max_N=300,
        n_bins=5,
        seed=0,
    )
    np.testing.assert_array_equal(
        result["subsample_indices"], np.array(random.Random(0).sample(range(5000), k=300))
    )
    assert result["embeddings"].shape[0] == 5000


# -------------------------------------------------------------------- configs --


@pytest.mark.parametrize("name", REEB_CONFIGS)
def test_configs_have_cap(name):
    path = CONFIG_DIR / name
    assert path.is_file()
    cfg = yaml.safe_load(path.read_text())
    assert cfg["max_N"] == 5000
    assert cfg["random_state"] == "${seed}"
    assert "random_seed" not in cfg


def test_cli_override():
    proc = subprocess.run(
        [
            sys.executable, "-m", "manylatents.main",
            "algorithms/latent=reeb_graph", "data=swissroll",
            "algorithms.latent.max_N=300", "--cfg", "job",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    cfg = yaml.safe_load(proc.stdout)
    assert cfg["algorithms"]["latent"]["max_N"] == 300
