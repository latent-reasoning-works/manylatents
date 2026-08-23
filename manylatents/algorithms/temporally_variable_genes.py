"""Temporally variable gene (TVG) selection.

Ranks genes by how much their expression moves across a trajectory and returns
the top ``n_top_genes``. This is the *temporal* counterpart to the *static*
``manylatents.singlecell.preprocessing.highly_variable_genes`` (scanpy-backed,
variance across cells) in the manylatents-omics extension.

Lives in ``algorithms/`` rather than ``metrics/`` because it is a feature
*selection* routine, not an evaluation metric: it returns gene selections, not
a score, and does not satisfy the ``Metric`` protocol.

Flavors mirror ``scanpy.pp.highly_variable_genes`` gene-for-gene, with the
*observation* axis being **time** rather than cells -- a trajectory is fed in
as if each timepoint were a cell:

``seurat``
    Satija et al. dispersion ranking. Expects log1p-transformed expression;
    the data are ``expm1``-ed back, per-gene dispersion ``var / mean`` is
    computed, genes are binned by mean expression and the dispersion is
    z-scored *within its bin*. Ranks by that normalized dispersion.
``seurat_v3``
    Stuart et al. variance-stabilizing ranking. Expects raw counts; a loess
    curve of ``log10(var)`` on ``log10(mean)`` gives a regularized standard
    deviation, the data are standardized by it and clipped at ``sqrt(n_obs)``,
    and genes are ranked by the resulting normalized variance. Needs the
    optional ``scikit-misc`` dependency (as scanpy does).

Sharing scanpy's vocabulary is deliberate: the omics ``highly_variable_genes``
adapter can import core, so core is the only place the two can converge.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["temporally_variable_genes", "TVG_FLAVORS"]

#: Supported ranking flavors, mirroring ``scanpy.pp.highly_variable_genes``.
TVG_FLAVORS = ("seurat", "seurat_v3")


def _seurat_dispersions_norm(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin-normalized log dispersion per gene, as scanpy's ``flavor="seurat"``.

    ``x`` is ``(n_obs, n_genes)`` log1p expression (observations = timepoints).
    """
    with np.errstate(over="ignore"):
        x = np.expm1(x)

    mean = x.mean(axis=0)
    var = x.var(axis=0, ddof=1)  # scanpy: mean_var(..., correction=1)

    mean[mean == 0] = 1e-12  # scanpy: set entries equal to zero to a small value
    with np.errstate(divide="ignore", invalid="ignore"):
        dispersion = var / mean
        # Logarithmized mean/dispersion, as in Seurat. A zero (or negative --
        # only reachable on non-count input) dispersion has no log, so it drops
        # out as NaN and lands at the bottom of the ranking.
        dispersion[dispersion == 0] = np.nan
        dispersion = np.log(dispersion)
        mean = np.log1p(mean)

    df = pd.DataFrame({"means": mean, "dispersions": dispersion})
    df["mean_bin"] = pd.cut(df["means"], bins=n_bins)

    grouped = df.groupby("mean_bin", observed=True)["dispersions"]
    bin_stats = grouped.agg(avg="mean", dev="std")

    # Genes alone in their bin have no within-bin std; scanpy gives them a
    # normalized dispersion of exactly 1 by setting dev := avg, avg := 0.
    one_gene_per_bin = bin_stats["dev"].isnull()
    if one_gene_per_bin.any():
        bin_stats.loc[one_gene_per_bin, "dev"] = bin_stats.loc[one_gene_per_bin, "avg"]
        bin_stats.loc[one_gene_per_bin, "avg"] = 0

    aligned = bin_stats.loc[df["mean_bin"]].set_index(df.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = (df["dispersions"] - aligned["avg"]) / aligned["dev"]
    return norm.to_numpy(dtype=float)


def _seurat_v3_variances_norm(x: np.ndarray, span: float) -> np.ndarray:
    """Loess-regularized normalized variance per gene, as ``flavor="seurat_v3"``.

    ``x`` is ``(n_obs, n_genes)`` raw counts (observations = timepoints).
    """
    try:
        from skmisc.loess import loess
    except ImportError as e:  # pragma: no cover - mirrors scanpy's message
        raise ImportError(
            'flavor="seurat_v3" needs the scikit-misc package; '
            "install it via `pip install scikit-misc`."
        ) from e

    n_obs, n_genes = x.shape
    mean = x.mean(axis=0)
    var = x.var(axis=0, ddof=1)

    # Constant genes have no log10(var); scanpy leaves their fitted variance at
    # 0 (reg_std == 1), which makes their normalized variance come out as 0.
    not_const = var > 0
    n_fit = int(not_const.sum())
    if n_fit < 3:
        # skmisc's loess segfaults -- not raises -- on a degree-2 fit with fewer
        # than 3 points, taking the interpreter with it, so this is a hard stop.
        raise ValueError(
            f'flavor="seurat_v3" fits a loess mean-variance trend across genes '
            f"and needs at least 3 non-constant genes; got {n_fit}. "
            'Use flavor="seurat" for a handful of genes.'
        )

    estimat_var = np.zeros(n_genes, dtype=np.float64)
    model = loess(
        np.log10(mean[not_const]), np.log10(var[not_const]), span=span, degree=2
    )
    try:
        model.fit()
    except ValueError as e:
        raise ValueError(
            f"loess mean-variance fit failed on {n_fit} non-constant genes at "
            f"span={span}: the local window holds too few genes. Raise `span` "
            'or use flavor="seurat".'
        ) from e
    estimat_var[not_const] = model.outputs.fitted_values
    reg_std = np.sqrt(10**estimat_var)

    # Clip the standardized values at sqrt(n_obs), as Seurat does.
    clip_val = reg_std * np.sqrt(n_obs) + mean
    clipped = np.minimum(x.astype(np.float64), clip_val)
    squared_sum = np.square(clipped).sum(axis=0)
    counts_sum = clipped.sum(axis=0)

    return (1 / ((n_obs - 1) * np.square(reg_std))) * (
        (n_obs * np.square(mean)) + squared_sum - 2 * counts_sum * mean
    )


def temporally_variable_genes(
    expr: np.ndarray,  # (n_genes, n_obs) -- genes x observations (e.g. timepoints)
    gene_names: Sequence,
    n_top_genes: int = 2000,
    flavor: str = "seurat",
    n_bins: int = 20,
    span: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(selected_positions_into_gene_names, selected_names)``, rank-ordered.

    Genes are scored across the observation axis (``axis=1`` of ``expr``, i.e.
    time) and the top ``n_top_genes`` are returned, most-variable first.

    Parameters
    ----------
    expr : np.ndarray
        ``(n_genes, n_obs)`` expression matrix. Note this is the *transpose* of
        scanpy's ``AnnData.X`` convention: rows are genes, columns are the
        observations (timepoints) that play the role of cells.
    gene_names : sequence
        One name per row of ``expr``.
    n_top_genes : int
        How many genes to return. Clipped to ``n_genes``.
    flavor : {"seurat", "seurat_v3"}
        Ranking statistic, matching the scanpy flavor of the same name.
        ``"seurat"`` ranks by bin-normalized dispersion and expects
        log1p-transformed expression; ``"seurat_v3"`` ranks by loess-regularized
        normalized variance and expects raw counts (and needs ``scikit-misc``).
    n_bins : int
        ``flavor="seurat"`` only: number of mean-expression bins the dispersion
        is normalized within.
    span : float
        ``flavor="seurat_v3"`` only: loess span used to fit the mean-variance
        trend.

    Notes
    -----
    Genes whose score is undefined (a constant gene under ``"seurat"``, say)
    score as ``NaN`` and are ranked last, so they are only ever returned when
    ``n_top_genes`` exceeds the number of well-defined genes.
    """
    if expr.ndim != 2:
        raise ValueError(f"Expected a single 2D trajectory matrix, got shape {expr.shape}.")

    if expr.shape[0] != len(gene_names):
        raise ValueError(
            f"Gene names count ({len(gene_names)}) does not match "
            f"gene dimension in expr {expr.shape[0]}."
        )

    if flavor not in TVG_FLAVORS:
        raise ValueError(f"flavor must be one of {TVG_FLAVORS}; got {flavor!r}.")

    if expr.shape[1] < 2:
        raise ValueError(
            f"Need at least 2 observations to rank temporal variability; "
            f"got {expr.shape[1]}."
        )

    # scanpy works in (n_obs, n_genes); we take (n_genes, n_obs).
    x = np.asarray(expr, dtype=np.float64).T

    if flavor == "seurat":
        score = _seurat_dispersions_norm(x, n_bins=n_bins)
    else:
        score = _seurat_v3_variances_norm(x, span=span)

    # Descending by score, NaN last (scanpy: `nan_to_num(..., nan=-inf)`).
    # Stable so ties keep gene order, matching scanpy's tie handling.
    ranked = np.argsort(-np.nan_to_num(score, nan=-np.inf), kind="stable")

    top_k = min(n_top_genes, expr.shape[0])
    top_idx = ranked[:top_k]
    tvg = [gene_names[idx] for idx in top_idx]

    return top_idx, np.asarray(tvg, dtype=object)
