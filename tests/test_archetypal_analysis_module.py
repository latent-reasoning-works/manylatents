"""Tests for ArchetypalAnalysisModule against the archetypes 0.6 API.

A successful fit emits scipy ``nnls`` RuntimeWarning lines; do not run these
tests with ``-W error``.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

archetypes = pytest.importorskip("archetypes")

from manylatents.algorithms.latent.aa import ArchetypalAnalysisModule

REPO_ROOT = Path(__file__).resolve().parents[1]


def _x50():
    return np.random.default_rng(0).random((50, 5))


def _x15():
    # 15 <= 3 * 5: furthest_sum takes its kernel branch.
    return np.random.default_rng(0).random((15, 5))


def _blobs():
    """Three Gaussian blobs (180 x 30) reduced to 10 dimensions with PCA."""
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.6, (60, 30)) for c in (0, 5, 10)])
    return PCA(n_components=10).fit_transform(X)


def _assert_simplex_rows(out, n_rows, k):
    out = np.asarray(out)
    assert out.shape == (n_rows, k)
    assert (out >= 0).all()
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_fit_and_transform_succeed():
    x = _x50()
    m = ArchetypalAnalysisModule(n_components=3)
    m.fit(x)
    _assert_simplex_rows(m.transform(x), 50, 3)


def test_seeded_fits_are_deterministic():
    x = _x50()
    a = ArchetypalAnalysisModule(n_components=3, random_state=7)
    b = ArchetypalAnalysisModule(n_components=3, random_state=7)
    a.fit(x)
    b.fit(x)
    np.testing.assert_allclose(
        np.asarray(a.transform(x)), np.asarray(b.transform(x)), atol=1e-8
    )


def test_blobs_fit_with_three_archetypes():
    x = _blobs()
    assert x.shape == (180, 10)
    m = ArchetypalAnalysisModule(n_components=3, random_state=0)
    m.fit(x)
    out = np.asarray(m.transform(x))
    assert out.shape == (180, 3)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_fitted_model_is_the_private_subclass():
    m = ArchetypalAnalysisModule(n_components=3)
    m.fit(_x50())
    assert type(m.model).__name__ == "_ArchetypesAA"
    assert isinstance(m.model, archetypes.AA)


def test_core_import_does_not_import_archetypes():
    code = (
        "import sys, manylatents.algorithms.latent; "
        "print('archetypes' in sys.modules)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert proc.stdout.strip() == "False"


def test_kernel_branch_fits():
    x = _x15()
    m = ArchetypalAnalysisModule(n_components=3, random_state=7)
    m.fit(x)
    _assert_simplex_rows(m.transform(x), 15, 3)


@pytest.mark.skipif(hasattr(np, "mat"), reason="np.mat exists in this NumPy")
def test_kernel_branch_fails_without_init_override():
    """Documents why _ArchetypesAA overrides _init_coefs."""
    from sklearn.utils.validation import check_array

    class _ValidateOnly(archetypes.AA):
        def _validate_data(self, X, **kwargs):
            return check_array(X, **kwargs)

    model = _ValidateOnly(n_archetypes=3, max_iter=100, random_state=7)
    with pytest.raises(AttributeError, match="np.mat"):
        model.fit(_x15())


def test_method_argument_is_kept():
    m = ArchetypalAnalysisModule(n_components=3, method="mu")
    assert m.method == "mu"


def test_packaged_config_instantiates_and_fits():
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(
        REPO_ROOT / "manylatents" / "configs" / "algorithms" / "latent" / "aa.yaml"
    )
    cfg.n_components = 3
    cfg.random_state = 0
    m = instantiate(cfg)
    assert isinstance(m, ArchetypalAnalysisModule)
    assert m.method == "pgd"
    m.fit(_x50())


def test_pin_comment_names_the_06_api():
    text = (REPO_ROOT / "pyproject.toml").read_text()
    assert "aa.py targets the archetypes 0.6 API" in text
    assert "written against the 0.11" not in text
