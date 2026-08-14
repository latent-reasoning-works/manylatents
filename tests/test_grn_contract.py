"""Contract tests for the GRN operation's SHAPE.

These pin the structure around the Granger estimator so the maths can be swapped
underneath without re-deriving it: the input contract and its refusals, the
output triple and its two index relationships, and the time-axis provenance
declaration that has to travel with the graph.

Nothing here tests the estimator's maths — that is
``tests/test_cflows_granger.py``, which is unchanged.

Every numeric boundary asserted here was measured against the estimator first,
not derived from its source.
"""

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from manylatents.algorithms.grn import (  # noqa: E402
    MIN_DIFFERENCED_ROWS,
    TIME_AXIS_DERIVED,
    TIME_AXIS_KINDS,
    TIME_AXIS_MEASURED,
    DerivedTimeAxisError,
    GrnContractError,
    TimeAxisNotStatedError,
    TimeAxisTooShortError,
    check_gene_trajectory,
    check_grn_triple,
    granger_grn_from_expression,
    grn_provenance,
    min_timepoints,
    require_time_axis,
    select_genes,
    threshold_edges,
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def make_traj(T=300, n_genes=3, seed=0):
    """``[T, n_genes]`` integrated random walks; gene 0 drives gene 1 at lag 1."""
    rng = np.random.default_rng(seed)
    dr = rng.standard_normal(T)
    dc = np.zeros(T)
    dc[1:] = 0.9 * dr[:-1] + 0.3 * rng.standard_normal(T - 1)
    cols = [np.cumsum(dr), np.cumsum(dc)]
    cols += [np.cumsum(rng.standard_normal(T)) for _ in range(n_genes - 2)]
    return np.column_stack(cols)


def gene_names(n):
    return ["R", "T"] + [f"N{i}" for i in range(n - 2)]


# ========================================================================== #
# 1. TIME-AXIS PROVENANCE — the condition that travels with the graph
# ========================================================================== #
def test_time_axis_has_no_default_and_omitting_it_refuses():
    """'Not stated' is the case the verdict was written about, so it must not
    fall through to a number."""
    with pytest.raises(TimeAxisNotStatedError):
        require_time_axis(None)

    traj = make_traj()
    with pytest.raises(TimeAxisNotStatedError):
        granger_grn_from_expression(traj, gene_names(3), downsample=1)


def test_measured_time_axis_is_admissible():
    assert require_time_axis(TIME_AXIS_MEASURED) == "measured"


def test_derived_time_axis_is_refused_by_default():
    """A pseudotime computed from the data under test is the geomancer defect."""
    with pytest.raises(DerivedTimeAxisError):
        require_time_axis(TIME_AXIS_DERIVED)

    traj = make_traj()
    with pytest.raises(DerivedTimeAxisError):
        granger_grn_from_expression(
            traj, gene_names(3), time_axis="derived", downsample=1
        )


def test_derived_refusal_names_the_reason_not_just_the_rule():
    """The message has to carry why, or the next caller just flips the flag."""
    with pytest.raises(DerivedTimeAxisError) as exc:
        require_time_axis(TIME_AXIS_DERIVED)
    msg = str(exc.value)
    assert "computed from the data being tested" in msg
    assert "granger-verdict" in msg
    # the label-permutation null cannot catch this; the message must say so
    assert "never reads a label" in msg


def test_derived_can_be_forced_for_methods_work_and_is_marked_not_causal():
    """Forcing must stay possible (building the data-level null needs it), but
    the resulting graph must not be able to pass as a regulatory claim."""
    assert require_time_axis(TIME_AXIS_DERIVED, allow_derived=True) == "derived"

    traj = make_traj()
    edges, _node_ids, _weights, prov = granger_grn_from_expression(
        traj,
        gene_names(3),
        time_axis="derived",
        allow_derived=True,
        downsample=1,
        with_provenance=True,
    )
    assert edges.shape[0] > 0  # it does run
    assert any("NOT-CAUSAL" in p for p in prov)
    assert any("time_axis=derived" in p for p in prov)


def test_time_axis_vocabulary_is_closed():
    for bad in ("pseudotime", "MEASURED", "", "real", 0, True):
        with pytest.raises(GrnContractError):
            require_time_axis(bad)
    assert TIME_AXIS_KINDS == ("measured", "derived")


def test_provenance_leads_with_the_time_axis():
    """It is the one fact that decides whether the graph means anything, so it
    goes first rather than somewhere in the middle of the tuple."""
    prov = grn_provenance(time_axis="measured", n_genes=4, downsample=1)
    assert prov[0] == "granger:time_axis=measured"
    assert isinstance(prov, tuple)
    assert all(isinstance(p, str) for p in prov)  # SparseGraph.provenance is str-only


def test_measured_run_is_not_marked_not_causal():
    prov = grn_provenance(time_axis="measured")
    assert not any("NOT-CAUSAL" in p for p in prov)


# ========================================================================== #
# 2. INPUT CONTRACT — [T, n_genes] / [T, n_cells, n_genes], time axis FIRST
# ========================================================================== #
def test_accepts_2d_and_3d_with_time_first():
    traj2d = make_traj(T=120, n_genes=3)
    arr, _names = check_gene_trajectory(traj2d, gene_names(3), downsample=1)
    assert arr.shape == (120, 3)

    traj3d = np.repeat(traj2d[:, None, :], 4, axis=1)  # [T, n_cells, n_genes]
    arr3, _ = check_gene_trajectory(traj3d, gene_names(3), downsample=1)
    assert arr3.shape == (120, 3)
    # 3-D is averaged over the CELL axis (axis=1), not the time axis
    np.testing.assert_allclose(arr3, traj2d)


@pytest.mark.parametrize("shape", [(10,), (2, 3, 4, 5), ()])
def test_refuses_wrong_ndim(shape):
    with pytest.raises(GrnContractError, match="time axis FIRST|ndim"):
        check_gene_trajectory(np.zeros(shape), ["a", "b"], downsample=1)


def test_refuses_name_column_mismatch_in_both_directions():
    traj = make_traj(T=120, n_genes=3)
    with pytest.raises(GrnContractError, match="gene_names"):
        check_gene_trajectory(traj, ["a", "b"], downsample=1)  # too few
    with pytest.raises(GrnContractError, match="gene_names"):
        check_gene_trajectory(traj, ["a", "b", "c", "d"], downsample=1)  # too many


def test_refuses_duplicate_gene_names_and_names_the_duplicate():
    """Undiagnosable today: the estimator dies inside pandas with 'The truth
    value of a Series is ambiguous', naming neither the gene nor the problem."""
    traj = make_traj(T=120, n_genes=3)
    with pytest.raises(GrnContractError, match="unique") as exc:
        check_gene_trajectory(traj, ["a", "b", "a"], downsample=1)
    assert "'a'" in str(exc.value) or "a" in str(exc.value)


# --- the silent one -------------------------------------------------------- #
def test_min_timepoints_formula():
    """T >= MIN_DIFFERENCED_ROWS * downsample + 1. Measured against the
    estimator for downsample in (1, 2, 3, 5, 10, 20)."""
    assert MIN_DIFFERENCED_ROWS == 5
    for d in (1, 2, 3, 5, 10, 20):
        assert min_timepoints(d) == 5 * d + 1


@pytest.mark.parametrize("downsample", [1, 2, 10])
def test_refuses_time_axis_too_short_at_the_measured_boundary(downsample):
    need = min_timepoints(downsample)
    names = gene_names(2)

    with pytest.raises(TimeAxisTooShortError):
        check_gene_trajectory(make_traj(T=need - 1, n_genes=2), names, downsample=downsample)

    # one more timepoint and it is accepted
    arr, _ = check_gene_trajectory(make_traj(T=need, n_genes=2), names, downsample=downsample)
    assert arr.shape[0] == need


def test_too_short_would_otherwise_be_a_silent_empty_graph():
    """The refusal exists because the estimator swallows this: it catches the
    per-pair failure and leaves NaN, so a too-short trajectory returns an EMPTY
    graph with no error, which reads as 'no regulation found'.

    Measured here rather than asserted: the raw estimator is called directly to
    show the behaviour the guard is protecting against."""
    from manylatents.algorithms.cflows_granger import granger_grn

    traj = make_traj(T=50, n_genes=2)  # one short of min_timepoints(10) == 51
    edges, node_ids, _weights = granger_grn(traj, gene_names(2), downsample=10)
    assert edges.shape[0] == 0  # <- silent: no error, no warning, empty graph
    assert node_ids.shape[0] == 2  # nodes are there, so it does not look empty either

    # the shape refuses instead
    with pytest.raises(TimeAxisTooShortError, match="EMPTY graph"):
        granger_grn_from_expression(
            traj, gene_names(2), time_axis="measured", downsample=10
        )


def test_downsample_must_be_positive():
    with pytest.raises(GrnContractError):
        min_timepoints(0)
    with pytest.raises(GrnContractError):
        check_gene_trajectory(make_traj(), gene_names(3), downsample=0)


def test_time_axis_is_checked_before_the_input_contract():
    """Ordering matters: an unstated time axis is refused before any array work,
    so a caller who never declared it cannot get a shape error instead and
    conclude the declaration was optional."""
    with pytest.raises(TimeAxisNotStatedError):
        granger_grn_from_expression(np.zeros((3,)), ["a"], downsample=1)


# ========================================================================== #
# 3. OUTPUT CONTRACT — the SparseGraph triple and its index relationships
# ========================================================================== #
def valid_triple():
    edges = np.array([[0, 1], [1, 0], [1, 2]], dtype=int)
    node_ids = np.array([0, 1, 3], dtype=int)
    weights = np.array([1.0, -2.0, 0.5], dtype=float)
    return edges, node_ids, weights


def test_valid_triple_passes():
    e, n, w = check_grn_triple(*valid_triple(), n_genes=4)
    assert e.shape == (3, 2) and n.shape == (3,) and w.shape == (3,)


def test_dtypes_are_pinned():
    e, n, w = valid_triple()
    with pytest.raises(GrnContractError, match="integer"):
        check_grn_triple(e.astype(float), n, w)
    with pytest.raises(GrnContractError, match="integer"):
        check_grn_triple(e, n.astype(float), w)
    with pytest.raises(GrnContractError, match="floating"):
        check_grn_triple(e, n, w.astype(int))


def test_edges_index_into_node_ids_not_into_gene_names():
    """The easiest thing to get wrong, and it mislabels every node silently."""
    _e, n, _w = valid_triple()
    # 3 is a valid GENE index (node_ids contains it) but not a valid POSITION:
    # there are only 3 nodes, so positions run 0..2.
    bad = np.array([[0, 3]], dtype=int)
    with pytest.raises(GrnContractError, match="edges index into node_ids"):
        check_grn_triple(bad, n, np.array([1.0]))


def test_node_ids_index_into_gene_names():
    e, n, w = valid_triple()
    with pytest.raises(GrnContractError, match="node_ids index into gene_names"):
        check_grn_triple(e, n, w, n_genes=3)  # node_ids has 3, so n_genes must be >= 4
    check_grn_triple(e, n, w, n_genes=4)  # fine


def test_weights_align_one_to_one_with_edges():
    e, n, w = valid_triple()
    with pytest.raises(GrnContractError, match="align"):
        check_grn_triple(e, n, w[:-1])


def test_node_ids_must_be_unique_and_ascending():
    e, _, w = valid_triple()
    with pytest.raises(GrnContractError, match="unique"):
        check_grn_triple(e, np.array([0, 0, 1], dtype=int), w)
    with pytest.raises(GrnContractError, match="ascending"):
        check_grn_triple(e, np.array([3, 1, 0], dtype=int), w)


def test_no_self_loops():
    with pytest.raises(GrnContractError, match="self-loop"):
        check_grn_triple(
            np.array([[1, 1]], dtype=int),
            np.array([0, 1], dtype=int),
            np.array([1.0]),
        )


def test_shapes_are_pinned():
    _, n, w = valid_triple()
    with pytest.raises(GrnContractError, match=r"\(E, 2\)"):
        check_grn_triple(np.zeros((3, 3), dtype=int), n, w)
    with pytest.raises(GrnContractError, match="1-D"):
        check_grn_triple(np.zeros((0, 2), dtype=int), n.reshape(-1, 1), np.zeros(0))


def test_empty_graph_is_a_valid_triple():
    """Zero edges is a legitimate answer, not a contract violation."""
    check_grn_triple(
        np.zeros((0, 2), dtype=int), np.array([0, 1], dtype=int), np.zeros(0), n_genes=2
    )


def test_the_triple_the_verb_returns_satisfies_the_contract():
    traj = make_traj(T=300, n_genes=4)
    names = gene_names(4)
    edges, node_ids, weights = granger_grn_from_expression(
        traj, names, time_axis="measured", downsample=1
    )
    check_grn_triple(edges, node_ids, weights, n_genes=len(names))
    # dense over the non-self grid: G*(G-1)
    assert edges.shape[0] == 4 * 3
    assert node_ids.tolist() == [0, 1, 2, 3]


@pytest.mark.filterwarnings("ignore")
def test_triple_builds_a_real_sparsegraph():
    """The downstream contract, asserted against manykinds itself when present."""
    SparseGraph = pytest.importorskip("manykinds").SparseGraph
    traj = make_traj(T=300, n_genes=3)
    edges, node_ids, weights, prov = granger_grn_from_expression(
        traj, gene_names(3), time_axis="measured", downsample=1, with_provenance=True
    )
    graph = SparseGraph(
        edges=edges, node_ids=node_ids, edge_weights=weights, provenance=prov
    )
    graph.validate()
    assert graph.provenance[0] == "granger:time_axis=measured"


# ========================================================================== #
# 4. THE STUBS — defaults reproduce today's behaviour; the maths is absent
# ========================================================================== #
def test_select_genes_default_keeps_every_gene():
    names = np.asarray(gene_names(4), dtype=object)
    idx, sel = select_genes(make_traj(T=120, n_genes=4), names)
    assert idx.tolist() == [0, 1, 2, 3]
    assert [str(s) for s in sel] == gene_names(4)


def test_select_genes_maths_is_not_implemented():
    with pytest.raises(NotImplementedError, match="ASCENDING index"):
        select_genes(make_traj(T=120, n_genes=4), gene_names(4), n_top_genes=2)


def test_threshold_edges_default_keeps_every_edge():
    e, n, w = valid_triple()
    e2, _n2, w2 = threshold_edges(e, n, w)
    np.testing.assert_array_equal(e, e2)
    np.testing.assert_array_equal(w, w2)


@pytest.mark.parametrize("kwargs", [{"alpha": 0.05}, {"top_k": 2}, {"alpha": 0.05, "top_k": 2}])
def test_threshold_maths_is_not_implemented(kwargs):
    with pytest.raises(NotImplementedError, match="no thresholding rule"):
        threshold_edges(*valid_triple(), **kwargs)


def test_selection_and_explicit_regulators_cannot_both_be_set():
    with pytest.raises(GrnContractError, match="ambiguous"):
        granger_grn_from_expression(
            make_traj(T=300, n_genes=3),
            gene_names(3),
            time_axis="measured",
            downsample=1,
            regulators=["R"],
            n_top_genes=2,
        )


def test_explicit_regulators_and_targets_still_work():
    """Today's behaviour is preserved: node_ids is the sorted union of the named
    regulator/target gene indices, indexing the ORIGINAL gene list."""
    traj = make_traj(T=300, n_genes=4)
    names = gene_names(4)  # ["R", "T", "N0", "N1"]
    edges, node_ids, weights = granger_grn_from_expression(
        traj, names, time_axis="measured", downsample=1, regulators=["R"], targets=["T", "N1"]
    )
    assert node_ids.tolist() == [0, 1, 3]  # R=0, T=1, N1=3
    check_grn_triple(edges, node_ids, weights, n_genes=len(names))


# ========================================================================== #
# 5. DELEGATION — the shape does not touch the maths
# ========================================================================== #
def test_composed_verb_delegates_to_the_unmodified_estimator():
    """Same input, same numbers: the shape adds guards, not arithmetic."""
    from manylatents.algorithms.cflows_granger import granger_grn

    traj = make_traj(T=300, n_genes=3)
    names = gene_names(3)
    direct = granger_grn(traj, names, downsample=1)
    composed = granger_grn_from_expression(traj, names, time_axis="measured", downsample=1)

    np.testing.assert_array_equal(direct[0], composed[0])
    np.testing.assert_array_equal(direct[1], composed[1])
    np.testing.assert_allclose(direct[2], composed[2], rtol=0, atol=0)


def test_known_directed_edge_survives_the_shape():
    """The R -> T edge the estimator finds is still there, still directed. This
    is a plumbing check, not a re-test of the maths."""
    traj = make_traj(T=600, n_genes=3, seed=1)
    names = gene_names(3)
    edges, node_ids, weights = granger_grn_from_expression(
        traj, names, time_axis="measured", downsample=1
    )
    pos = {int(g): p for p, g in enumerate(node_ids)}
    by_dir = {(a, b): v for (a, b), v in zip(edges.tolist(), weights.tolist())}
    rt = by_dir[(pos[0], pos[1])]
    tr = by_dir[(pos[1], pos[0])]
    assert rt > 4.0
    assert abs(rt) > abs(tr)
