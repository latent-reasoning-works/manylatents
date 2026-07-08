"""Tests for the Cflows Granger-causality GRN estimator.

Correctness-critical facts under test:

* the exact statsmodels index path
  (``res[1][0]["ssr_chi2test"][1]`` for p, ``res[1][1][1].params[1]`` for the
  lagged-cause coefficient) validated against an INDEPENDENT OLS fit;
* the signed-score formula and its ``10*ln(2)`` cap;
* directionality (r -> c detected, c -> r not) and coefficient sign;
* the output contract that feeds ``manykinds.SparseGraph``.

Unit-test synthetic series are built in *differenced space* (a clean one-way
VAR) and then integrated with ``cumsum``, because ``granger_grn`` internally
first-differences its input; the internal ``diff`` therefore recovers the clean
VAR increments. ``downsample=1`` is used so the differenced series stays long
enough for statsmodels to fit (documented in the reference note).
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

pytest.importorskip("statsmodels")  # optional dep — skip module if unavailable (manylatents convention)
from statsmodels.regression.linear_model import OLS  # noqa: E402
from statsmodels.tsa.stattools import grangercausalitytests  # noqa: E402

from manylatents.algorithms.cflows_granger import (
    SIGNED_SCORE_CAP,
    granger_grn,
    granger_signed_score_matrix,
    signed_score,
)


# --------------------------------------------------------------------------- #
# synthetic data helpers
# --------------------------------------------------------------------------- #
def make_integrated_pair(coef, T=1500, seed=0, noise=1.0):
    """Return a ``[T, 2]`` array with columns (r, c).

    Differenced series form a clean one-way VAR:
        dr_t = white noise
        dc_t = coef * dr_{t-1} + eps_t
    so r -> c holds with coefficient sign == sign(coef) and c -> r does not.
    The observed series are the integrals (cumsum) of the differences, so that
    ``granger_grn``'s internal first difference recovers (dr, dc).
    """
    rng = np.random.default_rng(seed)
    dr = rng.standard_normal(T)
    eps = rng.standard_normal(T) * noise
    dc = np.zeros(T)
    for t in range(1, T):
        dc[t] = coef * dr[t - 1] + eps[t]
    r = np.cumsum(dr)
    c = np.cumsum(dc)
    return np.column_stack([r, c])


# --------------------------------------------------------------------------- #
# 1. statsmodels index path validated against an independent OLS
# --------------------------------------------------------------------------- #
def test_estimator_index_path_matches_independent_ols():
    """``res[1][0]["ssr_chi2test"][1]`` is the p-value and
    ``res[1][1][1].params[1]`` is the r_{t-1} coefficient of an independent OLS
    ``c_t ~ c_{t-1} + r_{t-1} + const``."""
    rng = np.random.default_rng(3)
    T = 500
    r = rng.standard_normal(T)
    eps = rng.standard_normal(T) * 0.5
    c = np.zeros(T)
    for t in range(1, T):
        c[t] = 0.7 * r[t - 1] + eps[t]

    # data[[c, r]] => statsmodels tests column 2 (r) Granger-causes column 1 (c).
    frame = pd.DataFrame({"c": c, "r": r})[["c", "r"]]
    res = grangercausalitytests(frame, maxlag=(1,), verbose=False)

    sm_p = res[1][0]["ssr_chi2test"][1]
    sm_coef = res[1][1][1].params[1]

    # --- independent lag matrix + OLS, built by hand ---------------------- #
    # exog order for the joint model is [c_{t-1}, r_{t-1}, const].
    y = c[1:]
    c_lag = c[:-1]
    r_lag = r[:-1]
    const = np.ones_like(y)
    X_joint = np.column_stack([c_lag, r_lag, const])
    ols_joint = OLS(y, X_joint).fit()
    X_own = np.column_stack([c_lag, const])
    ols_own = OLS(y, X_own).fit()

    # coefficient on r_{t-1} is params index 1 in both statsmodels and ours.
    manual_coef = ols_joint.params[1]
    assert sm_coef == pytest.approx(manual_coef, rel=1e-9, abs=1e-9)
    np.testing.assert_allclose(res[1][1][1].params, ols_joint.params, rtol=1e-9, atol=1e-9)

    # independently reproduce the ssr chi2 p-value.
    nobs = ols_joint.nobs
    chi2_stat = nobs * (ols_own.ssr - ols_joint.ssr) / ols_joint.ssr
    manual_p = stats.chi2.sf(chi2_stat, 1)  # dof == maxlag == 1
    assert sm_p == pytest.approx(manual_p, rel=1e-9, abs=1e-12)
    assert 0.0 <= sm_p <= 1.0


# --------------------------------------------------------------------------- #
# 2. synthetic causality + sign + directionality
# --------------------------------------------------------------------------- #
def test_positive_causality_and_direction():
    traj = make_integrated_pair(coef=0.8, seed=0)
    scores = granger_signed_score_matrix(traj, ["r", "c"], downsample=1)
    s_rc = scores.loc["r", "c"]  # r -> c
    s_cr = scores.loc["c", "r"]  # c -> r
    # r -> c: significant (large magnitude) and POSITIVE.
    assert s_rc > 4.0
    # c -> r: not significant (small magnitude) => directionality holds.
    assert abs(s_cr) < 3.0
    assert abs(s_rc) > abs(s_cr)


def test_negative_causality_sign():
    traj = make_integrated_pair(coef=-0.8, seed=1)
    scores = granger_signed_score_matrix(traj, ["r", "c"], downsample=1)
    s_rc = scores.loc["r", "c"]
    s_cr = scores.loc["c", "r"]
    # r -> c: significant and NEGATIVE (repression).
    assert s_rc < -4.0
    assert abs(s_cr) < 3.0
    assert abs(s_rc) > abs(s_cr)


# --------------------------------------------------------------------------- #
# 3. signed-score cap
# --------------------------------------------------------------------------- #
def test_signed_score_cap_constant():
    # -ln(0 + 2**-10) == 10 * ln(2).
    assert SIGNED_SCORE_CAP == pytest.approx(10.0 * np.log(2.0), abs=1e-12)


def test_signed_score_cap_not_exceeded():
    traj = make_integrated_pair(coef=0.8, seed=2)
    _, _, edge_weights = granger_grn(traj, ["r", "c"], downsample=1)
    assert np.abs(edge_weights).max() <= SIGNED_SCORE_CAP + 1e-6


def test_signed_score_formula_and_bound():
    # magnitude grows as p shrinks; sign follows coef; bounded by the cap.
    assert signed_score(0.0, 1.0) == pytest.approx(SIGNED_SCORE_CAP)
    assert signed_score(0.0, -1.0) == pytest.approx(-SIGNED_SCORE_CAP)
    assert abs(signed_score(0.5, 3.0)) < abs(signed_score(1e-6, 3.0))
    for p in np.linspace(0.0, 1.0, 25):
        assert abs(signed_score(p, 1.0)) <= SIGNED_SCORE_CAP + 1e-9


# --------------------------------------------------------------------------- #
# 4. output shape / validity for SparseGraph
# --------------------------------------------------------------------------- #
def test_output_contract_for_sparsegraph():
    traj = make_integrated_pair(coef=0.8, seed=0)
    gene_names = ["r", "c"]
    edges, node_ids, edge_weights = granger_grn(traj, gene_names, downsample=1)

    # edges: int (E, 2) indexing into node_ids.
    assert edges.ndim == 2 and edges.shape[1] == 2
    assert np.issubdtype(edges.dtype, np.integer)
    assert np.issubdtype(node_ids.dtype, np.integer)
    assert edges.min() >= 0
    assert edges.max() < len(node_ids)

    # edge_weights aligns with edges and is float.
    assert edge_weights.shape[0] == edges.shape[0]
    assert np.issubdtype(edge_weights.dtype, np.floating)

    # node set = both genes; no self-loops.
    assert set(node_ids.tolist()) == {0, 1}
    assert not any(a == b for a, b in edges.tolist())


def test_known_causal_edge_present_and_directed():
    traj = make_integrated_pair(coef=0.8, seed=0)
    gene_names = ["r", "c"]
    edges, node_ids, edge_weights = granger_grn(traj, gene_names, downsample=1)

    # map node position -> gene index -> gene name.
    idx_to_name = {i: n for i, n in enumerate(gene_names)}
    pos_to_name = {pos: idx_to_name[node_ids[pos]] for pos in range(len(node_ids))}

    weight_by_dir = {
        (pos_to_name[a], pos_to_name[b]): w
        for (a, b), w in zip(edges.tolist(), edge_weights.tolist())
    }

    # r -> c present, strong, positive.
    assert ("r", "c") in weight_by_dir
    assert weight_by_dir[("r", "c")] > 4.0
    # c -> r present (dense output) but near-zero => directed, not symmetrized.
    assert abs(weight_by_dir[("c", "r")]) < 3.0
    assert abs(weight_by_dir[("r", "c")]) > abs(weight_by_dir[("c", "r")])


def test_multi_gene_subset_indexing():
    """node_ids index into gene_names; a regulator/target subset stays consistent."""
    rng = np.random.default_rng(7)
    T = 800
    # 4 genes; gene 1 (r) drives gene 3 (c). genes 0, 2 are noise.
    dr = rng.standard_normal(T)
    dc = np.zeros(T)
    for t in range(1, T):
        dc[t] = 0.9 * dr[t - 1] + rng.standard_normal() * 0.5
    cols = [
        np.cumsum(rng.standard_normal(T)),  # gene 0 "g0"
        np.cumsum(dr),                      # gene 1 "r"
        np.cumsum(rng.standard_normal(T)),  # gene 2 "g2"
        np.cumsum(dc),                      # gene 3 "c"
    ]
    traj = np.column_stack(cols)
    gene_names = ["g0", "r", "g2", "c"]

    edges, node_ids, edge_weights = granger_grn(
        traj, gene_names, regulators=["r"], targets=["c", "g0"], downsample=1
    )
    # node set = union of regulators/targets gene indices, sorted.
    assert node_ids.tolist() == sorted({1, 3, 0})  # r=1, c=3, g0=0
    assert edges.max() < len(node_ids)
    # find r -> c edge and confirm it is the strong, positive one.
    pos_r = int(np.where(node_ids == 1)[0][0])
    pos_c = int(np.where(node_ids == 3)[0][0])
    rc = [w for (a, b), w in zip(edges.tolist(), edge_weights.tolist())
          if a == pos_r and b == pos_c]
    assert len(rc) == 1 and rc[0] > 4.0


# --------------------------------------------------------------------------- #
# 5. input-shape handling (mean over cells) and preprocessing knobs
# --------------------------------------------------------------------------- #
def test_accepts_3d_trajectory_mean_over_cells():
    traj2d = make_integrated_pair(coef=0.8, seed=0)  # [T, 2]
    T, G = traj2d.shape
    # replicate into [T, n_cells, n_genes]; cell-mean must recover traj2d.
    n_cells = 5
    traj3d = np.repeat(traj2d[:, None, :], n_cells, axis=1)
    s2 = granger_signed_score_matrix(traj2d, ["r", "c"], downsample=1)
    s3 = granger_signed_score_matrix(traj3d, ["r", "c"], downsample=1)
    np.testing.assert_allclose(s2.values, s3.values, equal_nan=True, rtol=1e-9, atol=1e-9)


def test_self_pairs_excluded_as_nan():
    traj = make_integrated_pair(coef=0.8, seed=0)
    scores = granger_signed_score_matrix(traj, ["r", "c"], downsample=1)
    assert np.isnan(scores.loc["r", "r"])
    assert np.isnan(scores.loc["c", "c"])
