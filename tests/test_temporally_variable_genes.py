"""Temporally-variable-gene selection and its wiring into the Granger GRN.

Two things are under test here:

* :func:`~manylatents.algorithms.temporally_variable_genes.temporally_variable_genes`
  as a selection routine -- that it actually ranks signal-carrying genes above
  filler, and that its positions and names describe the same genes;
* the *wiring* of that selection into
  :func:`~manylatents.algorithms.cflows_granger.granger_grn` via
  ``n_top_genes`` / ``flavor`` -- that ``flavor`` is required, forwarded, and
  that selection genuinely restricts the estimated network.

* gene-for-gene equivalence of each flavor against
  ``scanpy.pp.highly_variable_genes`` -- TVG scores genes across *time* where
  scanpy scores them across *cells*, so the bridge is to hand scanpy an
  ``AnnData`` whose "cells" are the trajectory's timepoints::

      traj : (n_genes, n_obs)   ->   AnnData(X=traj.T)

  Any disagreement is then a real difference in the statistic, not an axis
  convention. ``scanpy``/``anndata`` are dev-only deps, so those tests skip
  individually rather than gating the rest of this file.
"""

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from manylatents.algorithms.cflows_granger import (
    _to_gene_time_frame,
    granger_grn,
    select_variable_genes,
)
from manylatents.algorithms.temporally_variable_genes import (
    _seurat_dispersions_norm,
    _seurat_v3_variances_norm,
    temporally_variable_genes,
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


# Selection delegates to `temporally_variable_genes`, whose flavors mirror
# scanpy's HVG and therefore assume *count-scale* input (`seurat_v3`) or
# log1p-of-counts (`seurat`). The random walks above are unbounded and signed,
# so the fixtures here map them onto counts first.


def _as_counts(series):
    """Map an unbounded walk onto a 5..95 count sweep, preserving its shape."""
    span = series.max() - series.min()
    unit = (series - series.min()) / span
    return np.round(5.0 + 90.0 * unit)


def _count_traj(n_bg=0, T=400, seed=0):
    """``[T, n_genes]`` count trajectory; columns 0/1 are the causal pair r -> c.

    ``quiet`` and ``dead`` are flat in time (zero variance) and must never be
    selected. ``n_bg`` Poisson background genes at a spread of mean expression
    levels populate scanpy's mean-expression bins, which the `seurat` flavor
    normalizes dispersion within -- with only a handful of genes every bin holds
    one gene and the flavor degenerates, so selection tests pass ``n_bg > 0``.
    """
    rng = np.random.default_rng(seed + 1)
    pair = make_integrated_pair(0.8, T=T, seed=seed)
    live = [_as_counts(pair[:, 0]), _as_counts(pair[:, 1])]

    mus = np.exp(rng.uniform(np.log(2.0), np.log(100.0), size=n_bg))
    bg = [rng.poisson(m, size=T).astype(float) for m in mus]

    quiet = np.full(T, 30.0)
    dead = np.zeros(T)

    traj = np.column_stack(live + bg + [quiet, dead])
    names = ["r", "c"] + [f"bg{i}" for i in range(n_bg)] + ["quiet", "dead"]
    return traj, names


def _seurat_scale(traj, flavor):
    """`seurat` reads log1p expression, `seurat_v3` reads raw counts.

    Both sides of every scanpy comparison below go through this, so a mismatch
    can never be a scaling artifact.
    """
    x = np.asarray(traj, dtype=np.float64)
    return np.log1p(x) if flavor == "seurat" else x


def make_expr(n_noise=48, T=400, coupling=1.5, seed=0):
    """(n_genes, T) genes-x-trajectory matrix. Gene 0 = R, gene 1 = T (R->T, lag 1)."""
    rng = np.random.default_rng(seed)
    dR = rng.normal(size=T)
    dT = np.zeros(T)
    dT[1:] = coupling * dR[:-1] + 0.1 * rng.normal(size=T - 1)
    rows = [np.cumsum(dR), np.cumsum(dT)]
    # low-variance filler: must NOT be selected as highly variable
    rows += [0.01 * np.cumsum(rng.normal(size=T)) for _ in range(n_noise)]
    return np.asarray(rows), ["R", "T"] + [f"N{i}" for i in range(n_noise)]


FLAVORS = ["seurat", "seurat_v3"]


@pytest.fixture(scope="module")
def small_traj():
    """Four genes -- cheap enough for the full O(n^2) pairwise Granger grid."""
    return _count_traj(n_bg=0)


@pytest.fixture(scope="module")
def binned_traj():
    """Enough genes for the `seurat` flavor's mean-expression binning to work."""
    return _count_traj(n_bg=40, T=300)


# --------------------------------------------------------------------------- #
# scanpy bridge (dev-only deps -- every user of these skips individually)
# --------------------------------------------------------------------------- #
def make_count_trajectory(n_bg=120, n_obs=400, seed=0):
    """Return ``(traj, gene_names)`` with ``traj`` of shape ``(n_genes, n_obs)``.

    Gene-major, unlike `_count_traj` above, because this is what both TVG and
    the scanpy bridge take. Count-scale, so it is valid input for `seurat_v3`
    directly and for `seurat` after ``log1p``. Composition:

    * ``sweep0`` / ``sweep1`` -- smooth monotone-ish random walks rescaled to
      sweep 5..95 counts: genuinely *temporally* variable, hugely overdispersed;
    * ``bg{i}`` -- Poisson noise at a log-uniform spread of means, so the
      mean-expression bins scanpy normalizes within are actually populated;
    * ``quiet`` -- flat at 30 counts, ``dead`` -- flat at 0. Both degenerate:
      zero variance, so no dispersion. They must never outrank a real gene.
    """
    rng = np.random.default_rng(seed)

    walks = []
    for _ in range(2):
        w = np.cumsum(rng.standard_normal(n_obs))
        w = (w - w.min()) / (w.max() - w.min())
        walks.append(np.round(5 + 90 * w))

    mus = np.exp(rng.uniform(np.log(2.0), np.log(100.0), size=n_bg))
    bg = [rng.poisson(m, size=n_obs).astype(float) for m in mus]

    quiet = np.full(n_obs, 30.0)
    dead = np.zeros(n_obs)

    traj = np.vstack(walks + bg + [quiet, dead])
    names = ["sweep0", "sweep1"] + [f"bg{i}" for i in range(n_bg)] + ["quiet", "dead"]
    return traj, names


def _scanpy():
    """scanpy + anndata, or skip this test."""
    sc = pytest.importorskip("scanpy")
    ad = pytest.importorskip("anndata")
    return sc, ad


def scanpy_hvg(traj, names, n_top_genes, flavor):
    """scanpy's HVG table for the same trajectory, gene-indexed.

    ``(n_genes, n_obs)`` trajectory -> AnnData with timepoints as cells.
    """
    sc, ad = _scanpy()
    adata = ad.AnnData(_seurat_scale(traj, flavor).T)  # (n_obs, n_genes)
    adata.var_names = list(names)
    return sc.pp.highly_variable_genes(
        adata, n_top_genes=n_top_genes, flavor=flavor, inplace=False
    )


def tvg(traj, names, n_top_genes, flavor):
    _, selected = temporally_variable_genes(
        _seurat_scale(traj, flavor), names, n_top_genes=n_top_genes, flavor=flavor
    )
    return selected.tolist()


@pytest.fixture(scope="module")
def trajectory():
    """Many genes across many mean-expression bins -- what scanpy needs to be
    compared against without cutoff ties."""
    return make_count_trajectory()


# --------------------------------------------------------------------------- #
# 1. the selection picks the genes that actually move
# --------------------------------------------------------------------------- #
def test_tvg_selects_the_signal_carrying_genes():
    expr, names = make_expr()
    idx, sel = temporally_variable_genes(expr, names, n_top_genes=10)
    assert {"R", "T"} <= set(sel)
    assert len(idx) == 10 and len(set(idx.tolist())) == 10


@pytest.mark.parametrize("flavor", FLAVORS)
def test_select_variable_genes_never_picks_a_gene_that_is_flat_in_time(
    binned_traj, flavor
):
    if flavor == "seurat_v3":
        pytest.importorskip("skmisc")
    traj, names = binned_traj
    expr = _seurat_scale(traj, flavor)

    top = select_variable_genes(expr, names, 10, flavor, downsample=1)
    assert "quiet" not in top.tolist()
    assert "dead" not in top.tolist()

    # They are ranked last, not merely excluded from the top 10.
    ranked = select_variable_genes(expr, names, len(names), flavor, downsample=1)
    assert set(ranked.tolist()[-2:]) == {"quiet", "dead"}


@pytest.mark.parametrize("flavor", FLAVORS)
def test_n_top_genes_larger_than_gene_count_keeps_everything(binned_traj, flavor):
    if flavor == "seurat_v3":
        pytest.importorskip("skmisc")
    traj, names = binned_traj
    kept = select_variable_genes(
        _seurat_scale(traj, flavor), names, 999, flavor, downsample=1
    )
    assert sorted(kept.tolist()) == sorted(names)


# --------------------------------------------------------------------------- #
# 2. flavor plumbing
# --------------------------------------------------------------------------- #
def test_flavor_is_forwarded_not_swallowed(binned_traj):
    """The two flavors rank by different statistics, so they must disagree."""
    pytest.importorskip("skmisc")
    traj, names = binned_traj
    seurat = select_variable_genes(_seurat_scale(traj, "seurat"), names, 10, "seurat",
                                   downsample=1)
    v3 = select_variable_genes(traj, names, 10, "seurat_v3", downsample=1)
    assert seurat.tolist() != v3.tolist()


def test_select_variable_genes_rejects_an_unknown_flavor(small_traj):
    traj, names = small_traj
    with pytest.raises(ValueError, match="flavor must be one of"):
        select_variable_genes(traj, names, 2, "cell_ranger", downsample=1)


# --------------------------------------------------------------------------- #
# 3. selection restricts the GRN
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("flavor", FLAVORS)
def test_n_top_genes_restricts_the_grn_to_the_selected_genes(binned_traj, flavor):
    if flavor == "seurat_v3":
        pytest.importorskip("skmisc")
    traj, names = binned_traj
    expr = _seurat_scale(traj, flavor)

    selected = select_variable_genes(expr, names, 2, flavor, downsample=1)
    edges, node_ids, weights = granger_grn(
        expr, names, downsample=1, n_top_genes=2, flavor=flavor
    )

    # Nodes are exactly the selected genes (as positions into `names`).
    assert node_ids.tolist() == sorted(names.index(g) for g in selected.tolist())
    # Dense over the non-self 2x2 grid -> both directed edges present.
    assert edges.shape == (2, 2)
    assert weights.shape == (2,)
    assert edges.max() < len(node_ids)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_node_ids_index_the_original_gene_list_after_selection(binned_traj, flavor):
    """Selection must not renumber the graph.

    Selection narrows the candidate set, so any index computed against that
    subset is *not* an index into the caller's `gene_names` -- gene 40 of a
    filtered set is a different gene. Returning one unmapped would relabel every
    node while still producing a perfectly well-formed graph, which is the
    failure mode this pins.

    The gene order is reversed so the surviving genes sit at high original
    positions. Without that, the top-ranked genes can land at positions 0/1,
    where a renumbered graph is indistinguishable from a correct one and the
    assertion below holds for the wrong reason.
    """
    if flavor == "seurat_v3":
        pytest.importorskip("skmisc")
    traj, names = binned_traj
    traj, names = traj[:, ::-1], list(names)[::-1]
    expr = _seurat_scale(traj, flavor)

    selected = select_variable_genes(expr, names, 3, flavor, downsample=1).tolist()
    expected = sorted(names.index(g) for g in selected)
    # Guard the guard: if the selection sat at 0..2 this test could not fail.
    assert min(expected) > 3

    _, node_ids, _ = granger_grn(expr, names, downsample=1, n_top_genes=3, flavor=flavor)
    assert node_ids.tolist() == expected


def test_n_top_genes_intersects_rather_than_replaces_regulators(binned_traj):
    """Selection narrows a caller-supplied candidate set; it does not widen it."""
    pytest.importorskip("skmisc")
    traj, names = binned_traj
    selected = select_variable_genes(traj, names, 2, "seurat_v3", downsample=1).tolist()
    dropped = next(g for g in names if g not in selected)

    _, node_ids, _ = granger_grn(
        traj, names,
        regulators=[selected[0]], targets=[selected[1], dropped],
        downsample=1, n_top_genes=2, flavor="seurat_v3",
    )
    assert node_ids.tolist() == sorted(names.index(g) for g in selected)


# --------------------------------------------------------------------------- #
# 4. selection is on by default, ranked by `seurat`
# --------------------------------------------------------------------------- #
# `n_top_genes=2000` keeps every gene of an HVG-sized input while bounding the
# O(n_genes^2) pair grid on a whole-transcriptome one, and `flavor="seurat"`
# assumes the log1p scale a trajectory arrives on. Neither is a safe guess for
# raw counts -- that is what `flavor="seurat_v3"` is for.
def test_the_default_flavor_is_seurat(small_traj):
    """An unspecified flavor must rank exactly as an explicit `seurat`."""
    traj, names = small_traj
    a = granger_grn(traj, names, downsample=1, n_top_genes=2)
    b = granger_grn(traj, names, downsample=1, n_top_genes=2, flavor="seurat")
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_flavor_none_with_selection_on_is_an_error(small_traj):
    """`None` is not a ranking statistic; the default exists so it need not be
    passed, not so it can be blanked."""
    traj, names = small_traj
    with pytest.raises(ValueError, match="flavor=None is not a ranking statistic"):
        granger_grn(traj, names, downsample=1, n_top_genes=2, flavor=None)


def test_flavor_is_ignored_when_selection_is_off(small_traj):
    """With `n_top_genes=None` nothing reads `flavor`, so it cannot misrank --
    including the `seurat_v3` that would otherwise demand count-scale input."""
    traj, names = small_traj
    a = granger_grn(traj, names, downsample=1, n_top_genes=None)
    b = granger_grn(traj, names, downsample=1, n_top_genes=None, flavor="seurat_v3")
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_the_default_keeps_every_gene_of_a_small_input(small_traj):
    """Selection being on by default must not silently drop genes: 2000 is far
    above this gene count, so the default has to match no selection at all."""
    traj, names = small_traj
    assert len(names) < 2000
    a = granger_grn(traj, names, downsample=1)
    b = granger_grn(traj, names, downsample=1, n_top_genes=None)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


# --------------------------------------------------------------------------- #
# 5. selection + Granger, end to end
# --------------------------------------------------------------------------- #
# `granger_grn` takes the trajectory as [T, n_genes]; `make_expr` builds the
# gene-major (n_genes, T) matrix, so it is transposed on the way in.
def test_end_to_end_recovers_the_directed_edge():
    expr, names = make_expr()
    edges, node_ids, w = granger_grn(
        expr.T, names, n_top_genes=10, flavor="seurat", downsample=1
    )
    # node_ids index the ORIGINAL gene list -> R is 0, T is 1
    pos = {int(g): p for p, g in enumerate(node_ids)}
    weight = {(a, b): v for (a, b), v in zip(edges.tolist(), w.tolist())}
    assert weight[(pos[0], pos[1])] > 4.0                       # R -> T, strong, positive
    assert weight.get((pos[1], pos[0]), 0.0) < weight[(pos[0], pos[1])]   # directed

    from manykinds import SparseGraph
    SparseGraph(edges=edges, node_ids=node_ids, edge_weights=w)  # validates or raises

# code is commented out until we reach a consensus on the necessity of alpha
'''
def test_null_does_not_fill_the_graph():
    """All-independent random walks: alpha=0.05 must not return a dense graph."""
    rng = np.random.default_rng(7)
    G = 20
    expr = np.cumsum(rng.normal(size=(G, 400)), axis=1)
    edges, _, _ = granger_grn(
        expr.T, [f"g{i}" for i in range(G)],
        n_top_genes=G, flavor="seurat", downsample=1,
    )
    assert edges.shape[0] < 0.15 * G * (G - 1)   # dense output would be 1.0
'''

# --------------------------------------------------------------------------- #
# 6. the per-gene statistic itself matches scanpy's, gene for gene
# --------------------------------------------------------------------------- #
def test_seurat_dispersions_norm_matches_scanpy(trajectory):
    traj, names = trajectory
    ours = _seurat_dispersions_norm(_seurat_scale(traj, "seurat").T, n_bins=20)
    theirs = scanpy_hvg(traj, names, 20, "seurat")["dispersions_norm"].to_numpy()

    # Degenerate genes are NaN for us and NaN/absent for scanpy; compare the
    # well-defined ones exactly and check the degenerate set agrees.
    finite = np.isfinite(theirs)
    np.testing.assert_allclose(ours[finite], theirs[finite], rtol=1e-10, atol=1e-10)
    assert set(np.flatnonzero(~np.isfinite(ours))) == set(np.flatnonzero(~finite))


def test_seurat_v3_variances_norm_matches_scanpy(trajectory):
    pytest.importorskip("skmisc")
    traj, names = trajectory
    ours = _seurat_v3_variances_norm(_seurat_scale(traj, "seurat_v3").T, span=0.3)
    theirs = scanpy_hvg(traj, names, 20, "seurat_v3")["variances_norm"].to_numpy()
    np.testing.assert_allclose(ours, theirs, rtol=1e-8, atol=1e-8)


# --------------------------------------------------------------------------- #
# 7. the resulting gene selection -- and its order -- match scanpy's
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_top_genes", [2, 5, 20, 50])
def test_seurat_selection_matches_scanpy(trajectory, n_top_genes):
    traj, names = trajectory
    ours = tvg(traj, names, n_top_genes, "seurat")
    df = scanpy_hvg(traj, names, n_top_genes, "seurat")
    theirs = set(df.index[df["highly_variable"]])

    assert len(ours) == n_top_genes
    # scanpy selects via `dispersions_norm >= cutoff`, so ties at the cutoff make
    # it return *more* than n_top_genes; ours is always the exact top-k.
    assert set(ours) <= theirs
    if len(theirs) == n_top_genes:
        assert set(ours) == theirs


@pytest.mark.parametrize("n_top_genes", [2, 5, 20, 50])
def test_seurat_v3_selection_matches_scanpy(trajectory, n_top_genes):
    pytest.importorskip("skmisc")
    traj, names = trajectory
    ours = tvg(traj, names, n_top_genes, "seurat_v3")
    df = scanpy_hvg(traj, names, n_top_genes, "seurat_v3")
    theirs = set(df.index[df["highly_variable"]])

    assert len(ours) == n_top_genes
    assert set(ours) == theirs


def test_seurat_rank_order_matches_scanpy_dispersion_order(trajectory):
    traj, names = trajectory
    n_top = 30
    ours = tvg(traj, names, n_top, "seurat")
    df = scanpy_hvg(traj, names, n_top, "seurat")
    theirs = (
        df["dispersions_norm"]
        .sort_values(ascending=False, na_position="last")
        .index[:n_top]
        .tolist()
    )
    assert ours == theirs


def test_seurat_v3_rank_order_matches_scanpy_rank_column(trajectory):
    pytest.importorskip("skmisc")
    traj, names = trajectory
    n_top = 30
    ours = tvg(traj, names, n_top, "seurat_v3")
    df = scanpy_hvg(traj, names, n_top, "seurat_v3")
    # scanpy exposes the ranking directly for this flavor.
    theirs = df["highly_variable_rank"].sort_values().index[:n_top].tolist()
    assert ours == theirs


# --------------------------------------------------------------------------- #
# 8. input contract of `temporally_variable_genes` itself
# --------------------------------------------------------------------------- #
def test_seurat_v3_refuses_a_loess_fit_it_cannot_make(trajectory):
    """Below 3 non-constant genes skmisc's loess segfaults rather than raising,
    so the guard has to fire before we ever reach it."""
    pytest.importorskip("skmisc")
    traj, _ = trajectory
    tiny = np.vstack([traj[0], traj[1], np.zeros(traj.shape[1])])
    with pytest.raises(ValueError, match="at least 3 non-constant genes"):
        temporally_variable_genes(
            tiny, ["a", "b", "flat"], n_top_genes=2, flavor="seurat_v3"
        )


def test_gene_positions_index_back_into_gene_names(trajectory):
    """The returned positions and names must describe the same genes."""
    traj, names = trajectory
    idx, selected = temporally_variable_genes(
        _seurat_scale(traj, "seurat"), names, n_top_genes=15, flavor="seurat"
    )
    assert [names[i] for i in idx] == selected.tolist()


def test_expr_is_genes_by_observations_not_the_scanpy_orientation(trajectory):
    """A transposed matrix must be rejected, not silently scored the wrong way."""
    traj, names = trajectory
    with pytest.raises(ValueError, match="does not match"):
        temporally_variable_genes(traj.T, names, n_top_genes=5, flavor="seurat")


# --------------------------------------------------------------------------- #
# 9. trajectory shape contract: `[T, n_cells, n_genes]` input
# --------------------------------------------------------------------------- #
# Selection reads the *levels* of the mean-over-cells trajectory, so the cell
# axis has to collapse by averaging before anything else happens. Flattening it
# into extra timepoints would silently change what "temporally variable" means.
@pytest.fixture(scope="module")
def celled_traj(binned_traj):
    """`binned_traj` resampled into cells: `[T, n_cells, n_genes]` counts.

    Poisson draws around the `[T, n_genes]` trajectory, so the cell mean tracks
    it and every cell stays count-scale -- valid input for `seurat_v3` directly
    and for `seurat` after ``log1p``.
    """
    base, names = binned_traj
    rng = np.random.default_rng(7)
    T, n_genes = base.shape
    cells = rng.poisson(np.broadcast_to(base[:, None, :], (T, 8, n_genes)))
    return cells.astype(float), names


def test_a_three_d_trajectory_collapses_to_one_row_per_timepoint(celled_traj):
    """The cell axis is averaged away, not flattened into `T * n_cells` rows."""
    traj, names = celled_traj
    frame = _to_gene_time_frame(traj, names, downsample=1, difference=False)
    assert frame.shape == (traj.shape[0], traj.shape[2])
    np.testing.assert_allclose(frame.to_numpy(), traj.mean(axis=1))


@pytest.mark.parametrize("flavor", FLAVORS)
def test_selecting_on_cells_matches_selecting_on_their_mean(celled_traj, flavor):
    """`[T, n_cells, n_genes]` must rank exactly as its pre-averaged `[T, n_genes]`.

    The reduction happens on the scale it is handed, so the pre-averaged
    reference has to be averaged on that same scale -- for `seurat`,
    ``log1p`` then mean is not mean then ``log1p``.
    """
    if flavor == "seurat_v3":
        pytest.importorskip("skmisc")
    traj, names = celled_traj
    celled = _seurat_scale(traj, flavor)

    from_cells = select_variable_genes(celled, names, 10, flavor, downsample=1)
    from_mean = select_variable_genes(celled.mean(axis=1), names, 10, flavor, downsample=1)
    np.testing.assert_array_equal(from_cells, from_mean)


def test_granger_grn_takes_a_three_d_trajectory_end_to_end(celled_traj):
    """Selection *and* the pairwise fits both accept the celled trajectory."""
    traj, names = celled_traj
    expr = _seurat_scale(traj, "seurat")

    from_cells = granger_grn(expr, names, downsample=1, n_top_genes=3, flavor="seurat")
    from_mean = granger_grn(
        expr.mean(axis=1), names, downsample=1, n_top_genes=3, flavor="seurat"
    )
    for a, b in zip(from_cells, from_mean):
        np.testing.assert_array_equal(a, b)


def test_a_trajectory_with_too_many_axes_is_rejected(celled_traj):
    """Only the two documented layouts are accepted; anything else is an error,
    not a guess at which axis is time."""
    traj, names = celled_traj
    with pytest.raises(ValueError, match=r"\[T, n_genes\] or \[T, n_cells, n_genes\]"):
        select_variable_genes(traj[:, :, None, :], names, 2, "seurat", downsample=1)


# --------------------------------------------------------------------------- #
# 10. the flavors measure *relative* variability, not raw variance
# --------------------------------------------------------------------------- #
# This is the confound the flavors exist to undo. In count data the variance of
# a gene is mostly a restatement of its mean (Poisson gives ``var == mean``, and
# overdispersion only steepens that), so ranking on raw variance ranks on
# expression level -- it keeps loud genes that do nothing and drops quiet genes
# that do something. Both flavors correct for it, by different routes: `seurat`
# z-scores log dispersion (``var / mean``) *within a mean-expression bin*, and
# `seurat_v3` divides by a variance read off a loess fit of ``log10(var)`` on
# ``log10(mean)``. The tests below pit the two rankings against each other on
# data built so they must disagree.


def _spearman(a, b):
    """Rank correlation, computed from ranks so no scipy dependency is needed."""
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable")
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable")
    return float(np.corrcoef(ra, rb)[0, 1])


def _ranking(traj, names, flavor):
    """``{gene_name: rank}`` over the whole gene list, 0 = most variable."""
    idx, _ = temporally_variable_genes(
        _seurat_scale(traj, flavor), names, n_top_genes=len(names), flavor=flavor
    )
    return {names[p]: r for r, p in enumerate(idx)}


@pytest.fixture(scope="module")
def mean_confounded_traj():
    """``(n_genes, n_obs)`` counts holding two genes that the two rankings
    disagree about.

    * ``loud_flat`` -- Poisson at mean 300. Enormous raw variance (~300) and
      nothing whatsoever going on: it sits exactly on the mean-variance trend
      the background genes define.
    * ``quiet_swept`` -- Poisson around a 1 -> 10 ramp. Raw variance of ~13,
      some twenty times smaller than ``loud_flat``, but two-and-a-half times
      what a mean-5 gene is entitled to.

    The ``bg`` genes are pure Poisson across a log-uniform spread of means, so
    they define that trend and populate the bins `seurat` normalizes within.
    """
    rng = np.random.default_rng(0)
    n_bg, n_obs = 200, 300

    mus = np.exp(rng.uniform(np.log(2.0), np.log(400.0), size=n_bg))
    bg = [rng.poisson(m, size=n_obs).astype(float) for m in mus]

    loud_flat = rng.poisson(300.0, size=n_obs).astype(float)
    ramp = 1.0 + 9.0 * (np.arange(n_obs) / (n_obs - 1))
    quiet_swept = rng.poisson(ramp).astype(float)

    traj = np.vstack(bg + [loud_flat, quiet_swept])
    names = [f"bg{i}" for i in range(n_bg)] + ["loud_flat", "quiet_swept"]
    return traj, names


def test_raw_variance_alone_would_pick_the_wrong_gene(mean_confounded_traj):
    """The control: without a flavor, this data ranks exactly backwards.

    If this ever stops holding, the two tests below are passing on data that no
    longer poses the problem, and prove nothing.
    """
    traj, names = mean_confounded_traj
    var = traj.var(axis=1, ddof=1)
    mean = traj.mean(axis=1)
    loud, quiet = names.index("loud_flat"), names.index("quiet_swept")

    # The boring gene really does have the larger raw variance, by a wide margin.
    assert var[loud] > 20 * var[quiet]
    # ...while carrying far less of it *per unit expression*: `loud_flat` sits
    # near the Poisson dispersion of 1 that the background defines, `quiet_swept`
    # at roughly 2.4, so the ratio between them is about 2.
    assert var[quiet] / mean[quiet] > 1.5 * (var[loud] / mean[loud])

    ranked = np.argsort(-var, kind="stable").tolist()
    assert ranked.index(loud) < 20        # naive ranking keeps the boring gene
    assert ranked.index(quiet) > 100      # ...and throws the interesting one away

    # And that is not a quirk of these two genes: across the whole gene list,
    # raw variance is very nearly a restatement of mean expression.
    assert _spearman(mean, var) > 0.99


@pytest.mark.parametrize("flavor", FLAVORS)
def test_a_flavor_ranks_relative_variability_over_raw_variance(
    mean_confounded_traj, flavor
):
    """Both flavors reverse the raw-variance verdict on the pair above."""
    if flavor == "seurat_v3":
        pytest.importorskip("skmisc")
    traj, names = mean_confounded_traj
    rank = _ranking(traj, names, flavor)

    assert rank["quiet_swept"] < rank["loud_flat"]
    # Not merely ahead of it -- the most variable gene in the whole list.
    assert rank["quiet_swept"] == 0

    # So a top-k selection keeps the quiet gene that raw variance discarded.
    # `select_variable_genes` reads the observation-major [T, n_genes] layout.
    kept = select_variable_genes(
        _seurat_scale(traj, flavor).T, names, 10, flavor, downsample=1
    ).tolist()
    assert "quiet_swept" in kept


@pytest.mark.parametrize("flavor", FLAVORS)
def test_a_flavor_ranking_is_decoupled_from_expression_level(
    mean_confounded_traj, flavor
):
    """The population-level statement of the same thing.

    Raw variance ranks these genes almost exactly as their mean does
    (rho > 0.99, asserted above). A flavor's ranking has to be close to
    independent of the mean, or it is still an expression filter in disguise.
    """
    if flavor == "seurat_v3":
        pytest.importorskip("skmisc")
    traj, names = mean_confounded_traj
    rank = _ranking(traj, names, flavor)

    mean = traj.mean(axis=1)
    ranked_by_flavor = np.array([rank[n] for n in names], dtype=float)
    assert abs(_spearman(mean, ranked_by_flavor)) < 0.15
