"""ScoreJacobianID (Stanczuk normal-bundle estimator) correctness.

The core test uses an ANALYTIC score (a Gaussian pancake: m free directions of large variance,
D-m normal directions of ~zero variance under noise sigma), so the estimator is verified
independent of any neural training. score(x) = -Sigma^{-1} x, Sigma = diag(var+sigma^2 on the m
tangent dims, sigma^2 on the D-m normal dims); the score cloud spans the normal space, and the NB
estimator must recover m = #(vanishing singular values).
"""
import numpy as np
import pytest

from manylatents.metrics.score_jacobian_id import ScoreJacobianID, _largest_gap_cut


class _AnalyticPancake:
    """Fitted-module stand-in with an exact Gaussian-pancake score (no training)."""

    def __init__(self, D, m, var=25.0, sigma=0.05):
        self.D, self.m, self.var, self.sigma_min = D, m, var, sigma
        self._is_fitted = True
        tangent = np.array([var] * m + [0.0] * (D - m), np.float32)
        self._inv_cov = 1.0 / (tangent + sigma ** 2)         # diag Sigma^{-1}

    def _standardize(self, X):
        return np.asarray(X, np.float32)

    def score(self, Z, sigma, standardized=True):
        return -np.asarray(Z, np.float32) * self._inv_cov[None, :]


def test_largest_gap_cut_basic():
    # 3 big + 2 tiny singular values -> 3 significant (normal) directions
    sv = np.array([10.0, 9.0, 8.0, 1e-4, 1e-5])
    assert _largest_gap_cut(sv, floor=1e-2) == 3


@pytest.mark.parametrize("D,m", [(10, 2), (20, 5), (30, 8)])
def test_nb_estimator_recovers_dimension(D, m):
    mod = _AnalyticPancake(D, m)
    X = np.zeros((40, D), np.float32)                        # points on the manifold (normal dims = 0)
    X[:, :m] = np.random.default_rng(0).standard_normal((40, m)) * 5.0
    est = ScoreJacobianID(embeddings=X, module=mod, return_per_sample=True, seed=0)
    assert est.shape == (40,)
    assert abs(np.median(est) - m) <= 1, f"NB estimator median {np.median(est)} != true m {m}"


def test_requires_fitted_module():
    with pytest.raises(ValueError):
        ScoreJacobianID(embeddings=np.zeros((5, 4), np.float32), module=None)
