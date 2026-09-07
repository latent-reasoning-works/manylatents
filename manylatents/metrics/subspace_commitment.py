"""Cross-fitted projection-energy concentration with a caller-chosen null.

Fit a low-rank subspace per label group and score each sample against bases
fit on the opposite split half: ``T = mean(1 - H(w)/log K)``, where ``w`` is
its normalized projection energy across the K subspaces. This measures
concentration, not agreement with the sample's own label. Bases are fitted
on centered rows; projection energies use the supplied origin (raw rows).

There is NO default null. The method owner must supply ``null_policy``:

* ``"random_bases"`` asks how concentrated energies are relative to
  independently oriented subspaces with matching dimensions, ranks and group
  counts. It is NOT centred under label independence. It does not preserve
  covariance or overlap between fitted bases, so generic anisotropy does NOT
  cancel. Cross-fitting addresses fitting bias, not this null mismatch.
* ``"label_permutation"`` asks whether concentration exceeds that under
  exchangeable labels conditional on the fixed embeddings and group sizes.
  Each permutation reruns the WHOLE split/fit/score procedure, with the same
  auxiliary split randomness as the observed evaluation. Any rank selection
  must also run inside every evaluation (currently rank is caller-specified,
  with the existing SVD shape cap applied during each fit). The caller owns
  whether unrestricted label exchangeability is appropriate for their data.
  Upper-tail ``p_value = (1 + #{T_perm >= T_observed}) / (B + 1)``.
  Runtime is roughly B+1 complete basis-fitting evaluations, versus one fit
  evaluation plus cheap random draws for ``random_bases``.

Known failure of the random-basis label-independence interpretation: give
both groups the SAME four collinear points, ``[-2, -1, 1, 2]`` on the first
axis of R^2, rank=1, n_null=3, random_seed=0. Every fitted basis spans the same
line, so energies are equal and mean commitment is exactly 0. The reproduced
null mean is 0.10335686140609159 and excess -0.10335686140609159 (up to floating
roundoff), despite NO label-specific structure. Zero-padding to four and
eight dimensions gives excess about -0.413 and -0.634, respectively. The
permutation null on the minimal fixture gives baseline=excess=0 and p_value=1.
The previously reported -0.45007 did not reproduce; its original dimensions
were not supplied and it is not evidence for this fixture.

These are different scientific questions wearing one name. In the sibling
reasoning-geometry repository, experiments/analysis/74_class_commitment_dynamics.py
around lines 190-193 and notes/nizar-task-commitment-block.md lines 18-25 read
nonpositive random-basis excess as a shared-collapse signature. Under label
permutation that number means something different. This module does not
choose between those interpretations or silently rewrite the downstream
findings: the method owner must name the null.

The existing cohort rule is retained: groups with fewer than four samples
are excluded from fitting and scoring. Permutations preserve all group sizes
and rerun that rule; the identities of included samples can therefore change.
Missing policy, unavailable labels/cohorts, or undefined scores raise
MeasurementUnavailable instead of producing a plausible numeric sentinel.
"""
from typing import Optional

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.metrics.registry import register_metric
from manylatents.utils.exceptions import MeasurementUnavailable


def _extract_labels(dataset: Optional[object]) -> Optional[np.ndarray]:
    """Extract labels from dataset (mirrors silhouette's convention)."""
    if dataset is None:
        return None
    labels = getattr(dataset, "metadata", None)
    if labels is None and hasattr(dataset, "get_labels"):
        labels = dataset.get_labels()
    if labels is None:
        return None
    return np.asarray(labels)


def _fit_basis(X: np.ndarray, rank: int) -> np.ndarray:
    """Top-``rank`` right singular vectors of the mean-centered rows -> (d, r)."""
    Xc = X - X.mean(0)
    try:
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        G = Xc.T @ Xc
        w, V = np.linalg.eigh(G)
        Vt = V[:, np.argsort(w)[::-1]].T
    return Vt[: min(rank, Vt.shape[0])].T


def group_half_bases(
    embeddings: np.ndarray, labels: np.ndarray, rank: int, rng: np.random.Generator
) -> tuple:
    """Split each group's samples in half and fit a basis per (group, half).

    Returns (bases_by_half, half_of_sample) where bases_by_half is
    ``{0: {lab: (d, r)}, 1: {lab: (d, r)}}`` and half_of_sample maps each
    sample index to its half (-1 for samples of skipped groups). Groups with
    < 4 samples are skipped."""
    bases_by_half: dict = {0: {}, 1: {}}
    half_of_sample = np.full(len(labels), -1, dtype=int)
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        if len(idx) < 4:
            continue
        idx = rng.permutation(idx)
        halves = (idx[: len(idx) // 2], idx[len(idx) // 2:])
        for h, hidx in enumerate(halves):
            bases_by_half[h][lab] = _fit_basis(embeddings[hidx], rank)
            half_of_sample[hidx] = h
    return bases_by_half, half_of_sample


def commitment_profile(embeddings: np.ndarray, bases_list: list) -> np.ndarray:
    """Per-sample commitment over a list of (d, r) orthonormal bases."""
    K = len(bases_list)
    if K < 2:
        return np.full(embeddings.shape[0], np.nan)
    E = np.stack(
        [((embeddings @ B) ** 2).sum(axis=1) for B in bases_list], axis=0
    )  # (K, n)
    tot = E.sum(axis=0)
    W = E / np.maximum(tot, 1e-12)
    H = -(W * np.log(np.maximum(W, 1e-12))).sum(axis=0)
    c = 1.0 - H / np.log(K)
    return np.where(tot > 1e-12, c, np.nan)


def _finite_mean(values: np.ndarray, context: str) -> float:
    """Refuse undefined observations rather than silently dropping them."""
    if not np.all(np.isfinite(values)):
        raise MeasurementUnavailable(
            f"SubspaceCommitment: {context} has undefined projection energy scores."
        )
    return float(np.mean(values))


def _cross_fit(embeddings, labels, rank, rng):
    """One complete split/fit/score evaluation, shared by observed and permuted data."""
    bases_by_half, half_of_sample = group_half_bases(embeddings, labels, rank, rng)
    if len(bases_by_half[0]) < 2:
        raise MeasurementUnavailable(
            "SubspaceCommitment: requires at least 2 usable label groups "
            "with >= 4 samples each."
        )
    c = np.full(len(labels), np.nan)
    for h in (0, 1):
        mask = half_of_sample == h
        other = list(bases_by_half[1 - h].values())
        c[mask] = commitment_profile(embeddings[mask], other)
    mean = _finite_mean(c[half_of_sample >= 0], "cross-fit")
    return mean, bases_by_half, half_of_sample


@register_metric(
    aliases=["subspace_commitment", "commitment"],
    default_params={"rank": 4, "n_null": 3, "random_seed": 0},
    description="Cross-fitted projection-energy concentration; caller must name "
    "null_policy='random_bases' or 'label_permutation'. See module for hypotheses.",
)
def SubspaceCommitment(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    rank: int = 4,
    n_null: int = 3,
    random_seed: int = 0,
    cache: Optional[dict] = None,
    null_policy: Optional[str] = None,
) -> dict:
    """Compute concentration relative to an explicitly named null.

    Args:
        embeddings: Finite (n_samples, n_features) embedding array. The supplied
            origin is used for scoring; fitting centers each group's half.
        dataset: Dataset with .metadata (or .get_labels()) giving aligned labels.
            Groups with fewer than four samples are excluded.
        module: LatentModule (unused).
        rank: Positive requested subspace rank; capped by each fit's SVD shape.
        n_null: Positive number B of null draws. For label_permutation this
            costs roughly B+1 COMPLETE basis-fitting evaluations.
        random_seed: Nonnegative seed; split randomness is reset identically
            for observed and every permuted evaluation.
        cache: Unused; fitted bases must not be reused across permutations.
        null_policy: REQUIRED caller choice (None raises MeasurementUnavailable).
            'random_bases' measures concentration relative to independently
            oriented subspaces, NOT excess centred under label independence.
            'label_permutation' tests exchangeable labels with fixed embeddings
            and group sizes, rerunning split/fit/score per draw. The module
            docstring gives the counterexample and downstream interpretation
            that the method owner must decide between. Example:
            SubspaceCommitment(X, dataset=ds, null_policy='label_permutation',
                               rank=1, n_null=99).

    Returns:
        dict: mean, null_mean, excess (mean - null_mean), n_groups, null_policy.
        Only label_permutation also reports the upper-tail p_value, counting
        ties with >= and using the (1 + count)/(B + 1) correction.

    Raises:
        MeasurementUnavailable: Missing/unknown null policy, missing labels,
            invalid inputs, fewer than two usable groups, undefined observed
            or null scores, or a failed basis decomposition. No null draw or
            scored observation is silently discarded.
    """
    if null_policy not in ("random_bases", "label_permutation"):
        raise MeasurementUnavailable(
            "SubspaceCommitment: missing or unknown null_policy; caller must "
            "choose 'random_bases' or 'label_permutation' (see module docstring)."
        )
    for name, value, minimum in (
        ("rank", rank, 1), ("n_null", n_null, 1), ("random_seed", random_seed, 0)
    ):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < minimum):
            raise MeasurementUnavailable(
                f"SubspaceCommitment: {name} must be an integer >= {minimum}."
            )
    embeddings = np.asarray(embeddings)
    if (embeddings.ndim != 2 or 0 in embeddings.shape
            or not np.issubdtype(embeddings.dtype, np.number)
            or np.iscomplexobj(embeddings) or not np.all(np.isfinite(embeddings))):
        raise MeasurementUnavailable("SubspaceCommitment: requires finite real 2-D embeddings.")
    labels = _extract_labels(dataset)
    if labels is None:
        raise MeasurementUnavailable("SubspaceCommitment: no labels available.")
    if labels.ndim != 1 or len(labels) != len(embeddings):
        raise MeasurementUnavailable("SubspaceCommitment: requires one aligned label per sample.")
    if any(lab is None or lab != lab for lab in labels):
        raise MeasurementUnavailable("SubspaceCommitment: missing label values.")

    rng = np.random.default_rng(random_seed)
    try:
        mean, bases_by_half, half_of_sample = _cross_fit(embeddings, labels, rank, rng)
        null_means = []
        if null_policy == "random_bases":
            # Keep the original contrast and RNG sequence when explicitly chosen.
            d = embeddings.shape[1]
            ranks = [B.shape[1] for B in bases_by_half[0].values()]
            scored = embeddings[half_of_sample >= 0]
            for _ in range(n_null):
                null_bases = [np.linalg.qr(rng.standard_normal((d, r)))[0] for r in ranks]
                null_means.append(_finite_mean(
                    commitment_profile(scored, null_bases), "random-bases null"
                ))
        else:
            # Separate label draws from the identically reset auxiliary fit RNG.
            permutation_rng = np.random.default_rng(random_seed)
            for _ in range(n_null):
                permuted_labels = permutation_rng.permutation(labels)
                perm_mean, _, _ = _cross_fit(
                    embeddings, permuted_labels, rank, np.random.default_rng(random_seed)
                )
                null_means.append(perm_mean)
    except np.linalg.LinAlgError as exc:
        raise MeasurementUnavailable("SubspaceCommitment: basis decomposition failed.") from exc

    null_mean = _finite_mean(np.asarray(null_means), "null")
    result = {
        "mean": mean,
        "null_mean": null_mean,
        "excess": mean - null_mean,
        "n_groups": len(bases_by_half[0]),
        "null_policy": null_policy,
    }
    if null_policy == "label_permutation":
        exceedances = np.count_nonzero(np.asarray(null_means) >= mean)
        result["p_value"] = float((1 + exceedances) / (n_null + 1))
    return result
