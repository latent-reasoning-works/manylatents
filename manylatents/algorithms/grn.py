"""Gene-regulatory-network operation over a gene x time matrix.

This module is the **shape** around the Granger estimator, not the estimator.
The maths of the test lives in :mod:`manylatents.algorithms.cflows_granger` and
is *delegated to* from here, unchanged. Two pieces of maths are **not** written
yet and are marked ``NotImplementedError`` with the signature and the contract
already pinned: gene selection (:func:`select_genes`) and edge thresholding
(:func:`threshold_edges`).

Why this is a separate module from ``cflows_granger``
-----------------------------------------------------
``granger_grn`` is not Cflows-specific: it takes a plain ``[T, n_genes]`` array
and runs ``statsmodels.tsa.stattools.grangercausalitytests`` underneath. No
neural ODE, no flow and no single cell enters it. ``Cflows.extra_outputs()`` is
therefore *one caller*, not the estimator's home, and other producers of a
gene x time matrix (real time-series bulk, pseudobulk over collection
timepoints, a decoded MIOFlow trajectory) are equally valid inputs. Keeping the
composed verb here means a second producer calls it rather than duplicating it.

What this module pins
---------------------
1. **The input contract** — :func:`check_gene_trajectory`. ``[T, n_genes]`` or
   ``[T, n_cells, n_genes]``, time axis FIRST, one name per gene, and the four
   refusals below.
2. **The output contract** — :func:`check_grn_triple`. The
   ``(edges, node_ids, edge_weights)`` triple that ``manykinds.SparseGraph``
   takes, with dtypes and the two index relationships pinned:
   ``edges`` indexes into ``node_ids``, ``node_ids`` indexes into ``gene_names``.
3. **The knobs** — plain keyword arguments here, mirrored as ``grn_*``
   *constructor* arguments on :class:`~manylatents.algorithms.lightning.cflows.Cflows`
   so a config can set them (``extra_outputs()`` takes no arguments, so a knob
   that is not a constructor argument cannot be reached on the flow path).
4. **The provenance of the time axis** — :func:`require_time_axis`. See below.

The condition that travels with the graph
-----------------------------------------
**Whether the trajectory's time axis was MEASURED or DERIVED is part of the
input contract, and there is no default.**

A Granger test cannot tell the two apart, and gets a confident p-value either
way. Geomancer deleted its own Granger step over exactly this: the step ordered
rows by a pseudotime computed *from the embedding*, then tested two coordinates
of that same embedding against each other in that order. Measured on null data
(``geomancer/docs/granger-verdict.md``, every number there marked ``[RAN]``):

* the shipped form rejected **100%** of the time (median p 9.4e-31), where two
  genuinely independent vectors under the *same* ordering rejected **13%** — so
  the ordering alone inflates about two-fold and the rest is self-reference;
* on pure noise it rejected in **12/12** seeds (median p 2.9e-29), and on a
  trajectory with *no* lead-lag it scored *more* significant (median p 9.9e-38)
  than on a true 20-step lead-lag (median p 5e-23);
* both directions reject, and it fails the time-reversal control 20/20.

A better statistic does not rescue it — on the identical fixture PCMCI+/ParCorr
is significant in *both* directions and CCM shows spurious convergence. The
defect is the data handed to the statistic, not the statistic.

It also cannot be caught by the usual null: a label-permutation twin is
bit-identical here, because the statistic never reads a label. Catching it needs
a **data-level** null pushed through the whole pipeline (embedding *and*
ordering *and* statistic). That null is not built here.

Hence the vocabulary, and hence the absence of a default:

``"measured"``
    The time axis carries information that did **not** come out of the
    expression matrix being tested — real collection times, hours post
    treatment, passage number, an experimental clock. Admissible.
``"derived"``
    The ordering was computed from the data being tested — a pseudotime, a
    diffusion ordering, the rank of an embedding coordinate. **Refused**, and
    :class:`DerivedTimeAxisError` says why. It can be forced with
    ``allow_derived=True`` for methods work (building the null, reproducing the
    verdict), and forcing it is recorded in the provenance so the resulting
    graph cannot be mistaken for a causal claim later.

Not stating it at all is also refused (:class:`TimeAxisNotStatedError`). "Not
stated" is the case the verdict was written about, so it is the one case that
must not fall through to a number.

A model-generated grid does not launder this. ``Cflows.extra_outputs()``
integrates over ``linspace(t_min, t_max, n_bins)`` where ``t_min``/``t_max``
come from the fit timepoints: that grid is a *reparametrisation* of the fit
times, and a reparametrisation cannot create measurement. So the declaration
belongs to whoever knows where those fit timepoints came from — which is the
person configuring the run, not the model. That is why ``grn_time_axis`` is a
constructor argument on ``Cflows`` with no default.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "TIME_AXIS_KINDS",
    "TIME_AXIS_MEASURED",
    "TIME_AXIS_DERIVED",
    "MIN_DIFFERENCED_ROWS",
    "GrnContractError",
    "TimeAxisNotStatedError",
    "DerivedTimeAxisError",
    "TimeAxisTooShortError",
    "require_time_axis",
    "min_timepoints",
    "check_gene_trajectory",
    "check_grn_triple",
    "select_genes",
    "threshold_edges",
    "grn_provenance",
    "granger_grn_from_expression",
]

# --------------------------------------------------------------------------- #
# Time-axis provenance vocabulary
# --------------------------------------------------------------------------- #
TIME_AXIS_MEASURED = "measured"
TIME_AXIS_DERIVED = "derived"
TIME_AXIS_KINDS = (TIME_AXIS_MEASURED, TIME_AXIS_DERIVED)

# Minimum rows AFTER downsampling and first-differencing that
# ``statsmodels.tsa.stattools.grangercausalitytests(..., maxlag=(1,))`` will
# fit. MEASURED, not guessed: at 4 rows statsmodels raises "Insufficient
# observations. Maximum allowable lag is 0"; at 5 it returns a p-value.
MIN_DIFFERENCED_ROWS = 5


class GrnContractError(ValueError):
    """Base class for the contract refusals in this module.

    Subclasses :class:`ValueError` so callers that already catch ``ValueError``
    around the estimator keep working.
    """


class TimeAxisNotStatedError(GrnContractError):
    """The caller did not say whether the time axis was measured or derived."""


class DerivedTimeAxisError(GrnContractError):
    """The time axis was derived from the data being tested."""


class TimeAxisTooShortError(GrnContractError):
    """Too few timepoints survive downsampling + differencing to fit the test."""


def require_time_axis(time_axis: Optional[str], *, allow_derived: bool = False) -> str:
    """Validate the time-axis provenance declaration. There is no default.

    Args:
        time_axis: ``"measured"`` or ``"derived"``. ``None`` is refused — see
            the module docstring for why "not stated" is the case that matters.
        allow_derived: Force a derived ordering through. For methods work only
            (building a data-level null, reproducing the verdict's numbers).
            The result is not a causal claim and :func:`grn_provenance` records
            that it was forced.

    Returns:
        The normalised provenance string.

    Raises:
        TimeAxisNotStatedError: ``time_axis`` is ``None``.
        DerivedTimeAxisError: ``time_axis`` is ``"derived"`` and
            ``allow_derived`` is False.
        GrnContractError: ``time_axis`` is outside the vocabulary.
    """
    if time_axis is None:
        raise TimeAxisNotStatedError(
            "time_axis must be stated as 'measured' or 'derived'; there is no default. "
            "'measured' = the ordering carries information from outside the expression "
            "matrix being tested (collection times, hours post treatment, passage "
            "number). 'derived' = the ordering was computed from that same matrix "
            "(a pseudotime, a diffusion ordering, the rank of an embedding coordinate). "
            "A Granger test cannot tell them apart and returns a confident p-value "
            "either way -- on pure noise it rejected 12/12 seeds. See "
            "geomancer/docs/granger-verdict.md."
        )
    if not isinstance(time_axis, str) or time_axis not in TIME_AXIS_KINDS:
        raise GrnContractError(
            f"time_axis must be one of {TIME_AXIS_KINDS}; got {time_axis!r}"
        )
    if time_axis == TIME_AXIS_DERIVED and not allow_derived:
        raise DerivedTimeAxisError(
            "time_axis='derived': the ordering was computed from the data being tested, "
            "so a Granger p-value on it is not evidence of direction. Measured on null "
            "data, this setup rejected 100% of the time (median p 9.4e-31) where "
            "independent vectors under the same ordering rejected 13%, and it rejected "
            "12/12 seeds on pure noise. A label-permutation null cannot catch it -- the "
            "statistic never reads a label -- so a data-level null through the whole "
            "pipeline is required, and is not built here. "
            "Pass allow_derived=True only for methods work (constructing that null); the "
            "result is recorded as non-causal in the provenance and must not be reported "
            "as a regulatory edge. See geomancer/docs/granger-verdict.md."
        )
    return time_axis


# --------------------------------------------------------------------------- #
# Input contract
# --------------------------------------------------------------------------- #
def min_timepoints(downsample: int) -> int:
    """Smallest ``T`` whose trajectory survives to a fittable test.

    The estimator takes ``[::downsample]`` along time and then first-differences,
    so ``ceil(T / downsample) - 1`` rows reach statsmodels and that must be at
    least :data:`MIN_DIFFERENCED_ROWS`. Solving gives
    ``T >= MIN_DIFFERENCED_ROWS * downsample + 1``.

    Verified against the estimator for ``downsample in (1, 2, 3, 5, 10, 20)``:
    one below this value yields an empty graph, this value yields edges.
    """
    if downsample < 1:
        raise GrnContractError(f"downsample must be >= 1; got {downsample}")
    return MIN_DIFFERENCED_ROWS * downsample + 1


def check_gene_trajectory(
    gene_traj,
    gene_names: Sequence,
    *,
    downsample: int = 10,
) -> tuple:
    """Validate a gene trajectory against the estimator's input contract.

    Args:
        gene_traj: ``[T, n_genes]`` or ``[T, n_cells, n_genes]``, **time axis
            first**. A 3-D input is averaged over the cell axis, exactly as the
            estimator does internally.
        gene_names: One name per gene column.
        downsample: The ``[::downsample]`` factor the estimator will apply.

    Returns:
        ``(arr, names)`` with ``arr`` the float ``[T, n_genes]`` matrix
        (cell-averaged if it arrived 3-D) and ``names`` a 1-D object array.

    Raises:
        GrnContractError: on any of the four refusals below.

    The refusals, and why each is here rather than left to the estimator:

    * **wrong ndim** — 1-D or 4-D+. The estimator already refuses this; it is
      re-checked so the composed verb fails before doing any work.
    * **name/column mismatch** — ``len(gene_names) != n_genes``. Also already
      refused by the estimator, and load-bearing: ``node_ids`` indexes into
      ``gene_names``, so a length mismatch silently corrupts that relationship.
    * **duplicate gene names** — currently NOT refused by the estimator. It
      builds a DataFrame with duplicate columns and dies inside pandas with
      "The truth value of a Series is ambiguous", which names neither the
      offending gene nor the real problem. Duplicate names also break the
      ``node_ids -> gene_names`` mapping, which is a lookup by name.
    * **time axis too short** — currently NOT refused, and silent. The estimator
      catches the per-pair failure and leaves ``NaN``, so a too-short trajectory
      returns an **empty graph with no error**. With the estimator's default
      ``downsample=10`` that swallows every trajectory shorter than 51
      timepoints: ``T=50`` returns zero edges and looks like "no regulation
      found". Measured, both branches.
    """
    arr = np.asarray(gene_traj, dtype=float)
    if arr.ndim == 3:
        arr = arr.mean(axis=1)
    elif arr.ndim != 2:
        raise GrnContractError(
            f"gene_traj must be [T, n_genes] or [T, n_cells, n_genes] with the time "
            f"axis FIRST; got shape {np.shape(gene_traj)} (ndim {np.ndim(gene_traj)})"
        )

    names = np.asarray(gene_names, dtype=object).reshape(-1)
    if arr.shape[1] != names.shape[0]:
        raise GrnContractError(
            f"gene_traj has {arr.shape[1]} genes but gene_names has {names.shape[0]}. "
            f"node_ids indexes into gene_names, so these must agree."
        )

    as_str = [str(n) for n in names.tolist()]
    if len(set(as_str)) != len(as_str):
        seen, dupes = set(), []
        for n in as_str:
            if n in seen and n not in dupes:
                dupes.append(n)
            seen.add(n)
        raise GrnContractError(
            f"gene_names must be unique; duplicates: {dupes[:5]}"
            f"{' ...' if len(dupes) > 5 else ''}. Regulators and targets are looked up "
            f"by name, so duplicates break the node_ids -> gene_names relationship."
        )

    needed = min_timepoints(downsample)
    if arr.shape[0] < needed:
        surviving = max(len(range(0, arr.shape[0], downsample)) - 1, 0)
        raise TimeAxisTooShortError(
            f"time axis too short: T={arr.shape[0]} with downsample={downsample} leaves "
            f"{surviving} row(s) after differencing, and the lag-1 Granger test needs at "
            f"least {MIN_DIFFERENCED_ROWS}. Need T >= {needed}. "
            f"(Without this check the estimator returns an EMPTY graph and no error, "
            f"which reads as 'no regulation found'.)"
        )
    return arr, names


# --------------------------------------------------------------------------- #
# Output contract
# --------------------------------------------------------------------------- #
def check_grn_triple(
    edges,
    node_ids,
    edge_weights,
    *,
    n_genes: Optional[int] = None,
) -> tuple:
    """Validate the ``(edges, node_ids, edge_weights)`` triple.

    This is the triple ``manykinds.SparseGraph`` takes. ``manykinds`` is behind
    the optional ``[cflows]`` extra, so this re-states its ``validate()`` rules
    without importing it, and adds the two index relationships that
    ``SparseGraph`` cannot check because it never sees ``gene_names``:

    * ``edges`` indexes into **``node_ids``** — every entry in ``[0, len(node_ids))``.
      Not into ``gene_names``, and this is the easiest thing to get wrong.
    * ``node_ids`` indexes into **``gene_names``** — every entry in
      ``[0, n_genes)``, unique and ascending.

    Args:
        edges: ``(E, 2)`` integer array of ``(regulator_pos, target_pos)``.
        node_ids: 1-D integer array of gene indices, sorted ascending.
        edge_weights: 1-D float array, one per edge.
        n_genes: ``len(gene_names)``. When given, the ``node_ids -> gene_names``
            bound is checked too.

    Returns:
        The triple, as ndarrays.

    Raises:
        GrnContractError: on any violation, naming which of the two index
            relationships broke.
    """
    edges = np.asarray(edges)
    node_ids = np.asarray(node_ids)
    edge_weights = np.asarray(edge_weights)

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise GrnContractError(f"edges must be (E, 2); got shape {edges.shape}")
    if not np.issubdtype(edges.dtype, np.integer):
        raise GrnContractError(
            f"edges must have integer dtype (SparseGraph.validate enforces this); "
            f"got {edges.dtype}"
        )
    if node_ids.ndim != 1:
        raise GrnContractError(f"node_ids must be 1-D; got shape {node_ids.shape}")
    if not np.issubdtype(node_ids.dtype, np.integer):
        raise GrnContractError(f"node_ids must have integer dtype; got {node_ids.dtype}")
    if edge_weights.ndim != 1:
        raise GrnContractError(
            f"edge_weights must be 1-D; got shape {edge_weights.shape}"
        )
    if not np.issubdtype(edge_weights.dtype, np.floating):
        raise GrnContractError(
            f"edge_weights must have floating dtype; got {edge_weights.dtype}"
        )
    if edge_weights.shape[0] != edges.shape[0]:
        raise GrnContractError(
            f"edge_weights has {edge_weights.shape[0]} entries but there are "
            f"{edges.shape[0]} edges; they must align one-to-one"
        )

    if node_ids.size and len(set(node_ids.tolist())) != node_ids.size:
        raise GrnContractError("node_ids must be unique")
    if node_ids.size and not np.all(np.diff(node_ids) > 0):
        raise GrnContractError("node_ids must be sorted ascending")

    if edges.size:
        if edges.min() < 0 or edges.max() >= node_ids.size:
            raise GrnContractError(
                f"edges index into node_ids (not into gene_names): every entry must lie "
                f"in [0, {node_ids.size}); got [{edges.min()}, {edges.max()}]"
            )
        self_loops = edges[edges[:, 0] == edges[:, 1]]
        if self_loops.size:
            raise GrnContractError(
                f"edges must not contain self-loops; got {len(self_loops)} "
                f"(a gene -> itself edge is collinear in the bivariate OLS)"
            )

    if n_genes is not None and node_ids.size and (
        node_ids.min() < 0 or node_ids.max() >= n_genes
    ):
        raise GrnContractError(
            f"node_ids index into gene_names: every entry must lie in "
            f"[0, {n_genes}); got [{node_ids.min()}, {node_ids.max()}]"
        )
    return edges, node_ids, edge_weights


# --------------------------------------------------------------------------- #
# The two pieces of maths that are NOT written yet
# --------------------------------------------------------------------------- #
def select_genes(
    gene_traj,
    gene_names: Sequence,
    *,
    n_top_genes: Optional[int] = None,
    flavor: str = "variance",
):
    """Choose which genes go into the pairwise test.

    **The selection maths is not implemented.** The default
    (``n_top_genes=None``) selects every gene, which is what the estimator does
    today, so the composed verb runs end to end with no behaviour change. Asking
    for a subset raises :class:`NotImplementedError`.

    Args:
        gene_traj: ``[T, n_genes]``, time axis first, already validated by
            :func:`check_gene_trajectory`.
        gene_names: One name per gene column.
        n_top_genes: How many genes to keep. ``None`` keeps all.
        flavor: Which selection rule to use. The vocabulary is open until the
            rule is chosen.

    Returns:
        ``(idx, names)`` — ``idx`` an ascending int array indexing into
        ``gene_names``, ``names`` the matching names. ``idx`` is what makes
        ``node_ids`` index the ORIGINAL gene list rather than a compacted one,
        so it must not be re-based.

    Raises:
        NotImplementedError: when ``n_top_genes`` is not ``None``.

    Two facts the body will need, both measured rather than assumed:

    * **Cost is quadratic in the gene count.** The estimator fits one bivariate
      OLS per ordered pair, serially: 4.5e-4 s/pair (G=10 -> 90 pairs -> 0.04s;
      G=20 -> 380 -> 0.18s; G=50 -> 2450 -> 1.11s). Selection is what makes the
      verb runnable at all, not a refinement of it.
    * **Constant genes are not dropped anywhere.** ``cflows_granger``'s module
      docstring says the reference pipeline drops genes whose mean trajectory
      has ``var == 0``; no such filter exists in the code. A constant gene
      survives as an isolated node in the graph with no edges. Whatever rule
      lands here subsumes that.

    ``scanpy`` cannot be imported from algorithm code — it is in the dev
    dependency-group only and will not be installed for users. A new runtime
    dependency goes in the ``[cflows]`` extra.
    """
    names = np.asarray(gene_names, dtype=object).reshape(-1)
    if n_top_genes is None:
        idx = np.arange(names.shape[0], dtype=int)
        return idx, names[idx]
    raise NotImplementedError(
        "select_genes: gene selection is not implemented; n_top_genes must be None "
        "(which keeps every gene). Return (idx, names) with idx an ASCENDING index "
        "into the original gene_names -- do not re-base it, node_ids is built from it "
        "and downstream reads node_ids against the original gene list. See the "
        "docstring for the measured cost curve and the constant-gene case."
    )


def threshold_edges(
    edges,
    node_ids,
    edge_weights,
    *,
    alpha: Optional[float] = None,
    top_k: Optional[int] = None,
):
    """Drop edges that are not worth keeping.

    **The thresholding rule is not implemented.** The default (both knobs
    ``None``) keeps every edge, which is the estimator's current dense
    ``G*(G-1)`` output, so the composed verb runs end to end with no behaviour
    change. Passing either knob raises :class:`NotImplementedError`.

    Args:
        edges: ``(E, 2)`` integer array indexing into ``node_ids``.
        node_ids: 1-D integer array indexing into ``gene_names``.
        edge_weights: 1-D float array, one per edge.
        alpha: Significance level to threshold at. ``None`` keeps all.
        top_k: Keep only the ``top_k`` strongest edges. ``None`` keeps all.

    Returns:
        The filtered ``(edges, node_ids, edge_weights)`` triple.

    Raises:
        NotImplementedError: when either knob is set.

    The fact the body needs, from ``cflows_granger``: the weight is
    ``sign(coef) * -ln(p + 2**-10)``, so sign is activation/repression and
    magnitude is monotone in ``-ln p``. Magnitude saturates at
    ``SIGNED_SCORE_CAP = -ln(2**-10) = 10*ln(2) ~= 6.9315``, reached at ``p = 0``,
    so the scale is bounded and a fixed magnitude cut is meaningful.

    Two things to decide rather than assume. Whether ``node_ids`` keeps genes
    that end up isolated after filtering, or is narrowed to the genes still
    carrying an edge — either is defensible, but :func:`check_grn_triple` will
    reject a narrowing that does not also re-base ``edges``, because ``edges``
    indexes into ``node_ids``. And whether ``alpha`` is corrected for the
    ``G*(G-1)`` tests being run, which at G=50 is 2450 of them.
    """
    edges, node_ids, edge_weights = check_grn_triple(edges, node_ids, edge_weights)
    if alpha is None and top_k is None:
        return edges, node_ids, edge_weights
    raise NotImplementedError(
        "threshold_edges: no thresholding rule is implemented; alpha and top_k must "
        "both be None (which keeps the dense G*(G-1) output). Re-base edges if you "
        "narrow node_ids -- edges indexes into node_ids, and check_grn_triple enforces "
        "it. See the docstring for the weight <-> p-value relation."
    )


# --------------------------------------------------------------------------- #
# Provenance — the record that travels with the graph
# --------------------------------------------------------------------------- #
def grn_provenance(
    *,
    time_axis: str,
    allow_derived: bool = False,
    n_genes: Optional[int] = None,
    n_selected: Optional[int] = None,
    n_top_genes: Optional[int] = None,
    flavor: Optional[str] = None,
    downsample: Optional[int] = None,
    alpha: Optional[float] = None,
    top_k: Optional[int] = None,
) -> tuple:
    """Build the provenance tuple that travels with the graph.

    ``manykinds.SparseGraph`` carries ``provenance: tuple[str, ...]``, and this
    is what goes in it. The first entry is always the time-axis declaration,
    because it is the one fact that decides whether the graph means anything.

    A graph produced from a forced derived ordering gets an explicit
    ``granger:NOT-CAUSAL`` marker, so that a graph which is only a methods
    artefact cannot later be read as a regulatory claim by someone who did not
    run it.
    """
    entries = [f"granger:time_axis={time_axis}"]
    if time_axis == TIME_AXIS_DERIVED and allow_derived:
        entries.append(
            "granger:NOT-CAUSAL=derived time axis forced; ordering computed from the "
            "data under test, no data-level null; not a regulatory claim"
        )
    if n_genes is not None:
        entries.append(f"granger:n_genes={n_genes}")
    if n_selected is not None:
        entries.append(f"granger:n_selected={n_selected}")
    if n_top_genes is not None:
        entries.append(f"granger:n_top_genes={n_top_genes}")
    if flavor is not None:
        entries.append(f"granger:flavor={flavor}")
    if downsample is not None:
        entries.append(f"granger:downsample={downsample}")
    if alpha is not None:
        entries.append(f"granger:alpha={alpha}")
    if top_k is not None:
        entries.append(f"granger:top_k={top_k}")
    return tuple(entries)


# --------------------------------------------------------------------------- #
# The composed verb
# --------------------------------------------------------------------------- #
def granger_grn_from_expression(
    gene_traj,
    gene_names: Sequence,
    *,
    time_axis: Optional[str] = None,
    allow_derived: bool = False,
    regulators: Optional[Sequence] = None,
    targets: Optional[Sequence] = None,
    n_top_genes: Optional[int] = None,
    flavor: str = "variance",
    downsample: int = 10,
    alpha: Optional[float] = None,
    top_k: Optional[int] = None,
    with_provenance: bool = False,
):
    """Gene trajectory -> selected genes -> Granger GRN -> validated triple.

    The estimator's maths is delegated, unchanged, to
    :func:`~manylatents.algorithms.cflows_granger.granger_grn`. The selection
    and thresholding steps are the stubs above: with their knobs left at the
    defaults this is exactly today's behaviour, guarded.

    Args:
        gene_traj: ``[T, n_genes]`` or ``[T, n_cells, n_genes]``, time first.
        gene_names: One name per gene column.
        time_axis: ``"measured"`` or ``"derived"``. **Required** — see
            :func:`require_time_axis`.
        allow_derived: Force a derived ordering through, for methods work.
        regulators: Candidate cause genes. ``None`` means every selected gene.
            Passed straight to the estimator, so this keeps working exactly as
            it does today. Cannot be combined with ``n_top_genes`` — naming the
            genes *and* asking for a selection over them is ambiguous, and the
            silent reading (selection wins, named genes vanish) is the one that
            loses data.
        targets: Candidate effect genes. Same rules as ``regulators``.
        n_top_genes: Gene-selection knob. ``None`` keeps every gene.
        flavor: Gene-selection rule.
        downsample: Estimator's ``[::downsample]`` factor along time.
        alpha: Thresholding knob. ``None`` keeps every edge.
        top_k: Thresholding knob. ``None`` keeps every edge.
        with_provenance: Return ``(edges, node_ids, edge_weights, provenance)``
            instead of the bare triple. Off by default so the return shape
            stays the triple every existing caller expects.

    Returns:
        ``(edges, node_ids, edge_weights)``, or that plus the provenance tuple
        when ``with_provenance`` is set. ``edges`` indexes into ``node_ids``;
        ``node_ids`` indexes into ``gene_names``.

    Raises:
        TimeAxisNotStatedError, DerivedTimeAxisError, TimeAxisTooShortError,
        GrnContractError: per the contract above, before any work is done.
        NotImplementedError: if a selection or thresholding knob is set.
    """
    time_axis = require_time_axis(time_axis, allow_derived=allow_derived)
    arr, names = check_gene_trajectory(gene_traj, gene_names, downsample=downsample)

    if n_top_genes is not None and (regulators is not None or targets is not None):
        raise GrnContractError(
            "regulators/targets and n_top_genes cannot both be set: naming the genes "
            "and selecting over them is ambiguous. Pass one or the other."
        )

    idx, selected = select_genes(arr, names, n_top_genes=n_top_genes, flavor=flavor)
    # Checked here rather than trusted, so a body swap in select_genes that re-bases
    # its index (the easy mistake) fails loudly instead of silently mislabelling every
    # node in the graph.
    idx = np.asarray(idx)
    if idx.ndim != 1 or (idx.size and not np.issubdtype(idx.dtype, np.integer)):
        raise GrnContractError(
            f"select_genes must return a 1-D integer index; got ndim {idx.ndim}, "
            f"dtype {idx.dtype}"
        )
    if idx.size and (idx.min() < 0 or idx.max() >= names.shape[0]):
        raise GrnContractError(
            f"select_genes must index into gene_names: entries must lie in "
            f"[0, {names.shape[0]}); got [{idx.min()}, {idx.max()}]. Do not re-base the "
            f"index onto the selected subset -- node_ids is read against the original "
            f"gene list."
        )
    if idx.size and not np.all(np.diff(idx) > 0):
        raise GrnContractError("select_genes must return a strictly ascending index")
    if np.asarray(selected, dtype=object).reshape(-1).shape[0] != idx.size:
        raise GrnContractError(
            f"select_genes returned {idx.size} indices but "
            f"{np.asarray(selected, dtype=object).reshape(-1).shape[0]} names"
        )
    selected_names = [str(n) for n in np.asarray(selected, dtype=object).reshape(-1)]

    from manylatents.algorithms.cflows_granger import granger_grn

    edges, node_ids, edge_weights = granger_grn(
        arr,
        [str(n) for n in names.tolist()],
        regulators=selected_names if regulators is None else [str(r) for r in regulators],
        targets=selected_names if targets is None else [str(t) for t in targets],
        downsample=downsample,
    )
    edges, node_ids, edge_weights = threshold_edges(
        edges, node_ids, edge_weights, alpha=alpha, top_k=top_k
    )
    edges, node_ids, edge_weights = check_grn_triple(
        edges, node_ids, edge_weights, n_genes=names.shape[0]
    )

    if not with_provenance:
        return edges, node_ids, edge_weights
    provenance = grn_provenance(
        time_axis=time_axis,
        allow_derived=allow_derived,
        n_genes=int(names.shape[0]),
        n_selected=int(node_ids.size),
        n_top_genes=n_top_genes,
        flavor=flavor,
        downsample=downsample,
        alpha=alpha,
        top_k=top_k,
    )
    return edges, node_ids, edge_weights, provenance
