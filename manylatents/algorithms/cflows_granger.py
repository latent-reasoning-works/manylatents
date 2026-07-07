"""Granger-causality gene regulatory network (GRN) estimator.

Ported from ``KrishnaswamyLab/cflows`` (``src/granger.py`` plus the signed-score
notebooks, e.g. ``notebooks/[FINAL]-47.1-granger-hvg-T.ipynb`` and its ``-A``
twin). This is the causal-network half of Cflows.

Given a gene-expression trajectory the estimator fits, for every ordered gene
pair ``(regulator r, target c)``, a bivariate first-order Granger-causality test
and reports a directed, signed, weighted edge ``r -> c``.

Reference algorithm (reproduced exactly):

* ``do_granger`` downsamples the trajectory ``::10`` along time, takes the
  first difference (``x - x.shift(1)``) for stationarity and drops NaNs.
* ``grangers_causation_matrix`` runs, per ordered pair, ::

      res = grangercausalitytests(data[[c, r]], maxlag=(1,), verbose=False)
      p    = res[1][0]["ssr_chi2test"][1]     # chi2 p-value
      coef = res[1][1][1].params[1]           # coef on lagged CAUSE r_{t-1}

  Per statsmodels' convention, ``data[[c, r]]`` tests whether *column 2* (``r``)
  Granger-causes *column 1* (``c``), i.e. the edge is ``r -> c``. In the
  unrestricted OLS ``c_t ~ c_{t-1} + r_{t-1} + const`` the exog order is
  ``['x1', 'x2', 'const']`` where ``x1 = c_{t-1}`` and ``x2 = r_{t-1}``, so
  ``params[1]`` is the coefficient on the lagged cause ``r_{t-1}``.
* The signed score (notebook cell) is ::

      log_pval  = -np.log(p + 2 ** -10)
      signed    = np.sign(coef) * log_pval

  hence ``signed_score(r->c) = sign(coef) * (-ln(p + 2**-10))`` whose maximum
  magnitude is ``-ln(2**-10) = 10 * ln(2) ~= 6.9315``.

The reference pipeline does **not** z-score / standardize the input; the only
notebook preprocessing beyond ``do_granger`` is dropping genes whose
mean-over-cells trajectory is constant in time (``var == 0``). No scaling is
applied here either.

Dependencies are kept light: numpy, pandas, statsmodels.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

__all__ = [
    "granger_grn",
    "granger_signed_score_matrix",
    "signed_score",
    "SIGNED_SCORE_CAP",
]

# Small floor added to the p-value so ``-ln(p)`` stays finite when p == 0.
# Matches the reference constant ``2 ** -10``.
_P_FLOOR = 2.0 ** -10

# Maximum magnitude of a signed score: reached when p == 0, i.e.
# -ln(0 + 2**-10) = 10 * ln(2) ~= 6.9315.
SIGNED_SCORE_CAP = -np.log(_P_FLOOR)  # == 10.0 * np.log(2.0)


def signed_score(p_value: float, coef: float) -> float:
    """Signed Granger score ``sign(coef) * (-ln(p + 2**-10))``.

    Positive when the lagged cause enters with a positive coefficient
    (activation), negative for a negative coefficient (repression); its
    magnitude grows as the p-value shrinks and is capped at
    :data:`SIGNED_SCORE_CAP`.
    """
    return float(np.sign(coef) * (-np.log(p_value + _P_FLOOR)))


def _to_gene_time_frame(
    gene_traj: np.ndarray,
    gene_names: Sequence,
    downsample: int,
) -> pd.DataFrame:
    """Return the preprocessed ``[T', genes]`` frame used by the Granger tests.

    Accepts ``gene_traj`` as ``[T, n_genes]`` (already mean over cells) or
    ``[T, n_cells, n_genes]`` (mean over the cell axis is taken first). The
    preprocessing mirrors ``do_granger``: downsample ``::downsample`` along
    time, first-difference along time, drop NaNs. Columns are ``gene_names``.
    """
    arr = np.asarray(gene_traj, dtype=float)
    if arr.ndim == 3:
        # [T, n_cells, n_genes] -> mean over the cell axis -> [T, n_genes].
        arr = arr.mean(axis=1)
    elif arr.ndim != 2:
        raise ValueError(
            f"gene_traj must be [T, n_genes] or [T, n_cells, n_genes]; got shape {arr.shape}"
        )

    gene_names = np.asarray(gene_names)
    if arr.shape[1] != gene_names.shape[0]:
        raise ValueError(
            f"gene_traj has {arr.shape[1]} genes but gene_names has {gene_names.shape[0]}"
        )
    if downsample < 1:
        raise ValueError(f"downsample must be >= 1; got {downsample}")

    frame = pd.DataFrame(arr, columns=list(gene_names))
    # Reference: trajs.T[::10] -> downsample every `downsample`-th timepoint.
    frame = frame.iloc[::downsample]
    # Reference: trajs - trajs.shift(1); dropna() -> first difference for stationarity.
    frame = frame.diff().dropna()
    return frame


def _pair_p_and_coef(frame: pd.DataFrame, cause: str, effect: str) -> tuple:
    """Run the bivariate Granger test for edge ``cause -> effect``.

    Builds ``data[[effect, cause]]`` so statsmodels tests whether column 2
    (``cause``) Granger-causes column 1 (``effect``). Returns ``(p, coef)`` with
    ``p`` the ssr chi2 p-value and ``coef`` the lagged-cause coefficient.
    """
    two_col = frame[[effect, cause]]
    with warnings.catch_warnings():
        # statsmodels emits Future/Value warnings around `verbose`; the
        # reference suppresses them. We mirror `verbose=False` exactly.
        warnings.simplefilter("ignore")
        res = grangercausalitytests(two_col, maxlag=(1,), verbose=False)
    p_value = res[1][0]["ssr_chi2test"][1]
    coef = res[1][1][1].params[1]
    return float(p_value), float(coef)


def granger_signed_score_matrix(
    gene_traj: np.ndarray,
    gene_names: Sequence,
    regulators: Optional[Sequence] = None,
    targets: Optional[Sequence] = None,
    downsample: int = 10,
) -> pd.DataFrame:
    """Signed Granger score for every ``(regulator, target)`` gene pair.

    Returns a DataFrame indexed by regulator gene name (rows = causes) with
    target gene names as columns (cols = effects); ``df.loc[r, c]`` is
    ``signed_score(r -> c)``. Self-pairs (``r == c``) are left as ``NaN``: the
    bivariate OLS would be perfectly collinear and a gene->itself edge is not a
    meaningful GRN edge. Degenerate pairs whose test fails are also ``NaN``.

    This mirrors the reference ``signed_score_df`` (index = regulators / ``in``,
    columns = targets / ``out``).
    """
    gene_names = np.asarray(gene_names)
    if regulators is None:
        regulators = list(gene_names)
    if targets is None:
        targets = list(gene_names)

    known = set(map(str, gene_names.tolist()))
    for group, label in ((regulators, "regulators"), (targets, "targets")):
        missing = [g for g in group if str(g) not in known]
        if missing:
            raise ValueError(f"{label} not found in gene_names: {missing[:5]}")

    frame = _to_gene_time_frame(gene_traj, gene_names, downsample)

    scores = pd.DataFrame(
        np.full((len(regulators), len(targets)), np.nan),
        index=list(regulators),
        columns=list(targets),
    )
    for r in regulators:
        for c in targets:
            if r == c:
                # Explicit self-loop exclusion (collinear / non-meaningful).
                continue
            try:
                p_value, coef = _pair_p_and_coef(frame, cause=r, effect=c)
            except Exception:
                # Degenerate pair (e.g. constant series); leave as NaN.
                continue
            scores.loc[r, c] = signed_score(p_value, coef)
    return scores


def granger_grn(
    gene_traj: np.ndarray,
    gene_names: Sequence,
    regulators: Optional[Sequence] = None,
    targets: Optional[Sequence] = None,
    downsample: int = 10,
):
    """Estimate a directed, signed, weighted Granger-causality GRN.

    Parameters
    ----------
    gene_traj : np.ndarray
        Gene trajectory, either ``[T, n_genes]`` (already mean over cells) or
        ``[T, n_cells, n_genes]`` (the cell axis is averaged first).
    gene_names : sequence
        Gene names, one per gene column; nodes of the GRN are genes.
    regulators, targets : sequence, optional
        Candidate cause / effect gene names. Default: all genes.
    downsample : int
        Time downsampling factor (reference uses 10). For short unit-test
        series pass ``downsample=1`` so the differenced series is long enough
        for statsmodels to fit.

    Returns
    -------
    edges : np.ndarray, shape (E, 2), int
        ``(regulator_pos, target_pos)`` pairs indexing **into** ``node_ids``.
    node_ids : np.ndarray, int
        Gene indices (into ``gene_names``) that form the GRN nodes; the union
        of ``regulators`` and ``targets``, sorted ascending.
    edge_weights : np.ndarray, shape (E,), float
        Signed score ``signed_score(r -> c)`` for each edge. Directed: the edge
        ``r -> c`` is kept as-is (never symmetrized).

    Notes
    -----
    The returned network is dense over the non-self ``regulators x targets``
    grid (thresholding/ranking is a downstream step in the reference), so a
    non-causal edge is *present with a near-zero weight* rather than absent.
    """
    gene_names = np.asarray(gene_names)
    if regulators is None:
        regulators = list(gene_names)
    if targets is None:
        targets = list(gene_names)

    scores = granger_signed_score_matrix(
        gene_traj, gene_names, regulators=regulators, targets=targets, downsample=downsample
    )

    name_to_idx = {str(name): i for i, name in enumerate(gene_names.tolist())}
    reg_idx = [name_to_idx[str(r)] for r in regulators]
    tgt_idx = [name_to_idx[str(c)] for c in targets]

    # Nodes = union of candidate regulators and targets (gene indices), sorted.
    node_ids = np.array(sorted(set(reg_idx) | set(tgt_idx)), dtype=int)
    pos_of = {gene_idx: pos for pos, gene_idx in enumerate(node_ids.tolist())}

    edges = []
    weights = []
    for r in regulators:
        r_pos = pos_of[name_to_idx[str(r)]]
        for c in targets:
            if r == c:
                continue
            w = scores.loc[r, c]
            if pd.isna(w):
                continue
            edges.append((r_pos, pos_of[name_to_idx[str(c)]]))
            weights.append(float(w))

    edges_arr = np.asarray(edges, dtype=int).reshape(-1, 2)
    weights_arr = np.asarray(weights, dtype=float)
    return edges_arr, node_ids, weights_arr
