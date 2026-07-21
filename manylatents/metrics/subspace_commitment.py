"""Subspace-commitment order parameter.

Given labeled embeddings, fit a low-rank subspace per label group and measure,
per sample, how concentrated the sample's projection energy is across the
group subspaces: ``commitment = 1 - H(w)/log K`` where ``w`` is the normalized
energy distribution over the K group subspaces. 1 = the sample lies in one
group's subspace (committed); 0 = spread evenly (uncommitted / wandering).

Reported with a rank/count-matched RANDOM-bases null computed identically
(``excess = mean - null_mean``), so generic geometry (anisotropy, norm
structure) cancels in the excess and only label-structured concentration
remains.

Scoring is SPLIT-HALF CROSS-FIT: each group's samples are split in two, bases
are fit per half, and every sample is scored against the bases fit on the
halves it did NOT contribute to. This removes in-sample fitting bias — on
isotropic data the excess is ~0 by construction, not by luck. Groups need at
least 4 samples to participate.
"""
import logging
import warnings
from typing import Optional

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.metrics.registry import register_metric

logger = logging.getLogger(__name__)


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
    Xc = np.nan_to_num(X - X.mean(0))
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


@register_metric(
    aliases=["subspace_commitment", "commitment"],
    default_params={"rank": 4, "n_null": 3, "random_seed": 0},
    description="Concentration of per-sample projection energy across per-label "
    "subspaces (1 = committed to one group, 0 = wandering), with a "
    "rank-matched random-bases null (excess).",
)
def SubspaceCommitment(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    rank: int = 4,
    n_null: int = 3,
    random_seed: int = 0,
    cache: Optional[dict] = None,
) -> dict:
    """Compute the subspace-commitment order parameter.

    Args:
        embeddings: (n_samples, n_features) embedding array.
        dataset: Dataset with .metadata (or .get_labels()) giving group labels.
        module: LatentModule (unused).
        rank: Rank of each group subspace.
        n_null: Random-bases draws for the null.
        random_seed: Seed for the null bases.

    Returns:
        dict: ``mean`` (mean per-sample commitment), ``null_mean`` (matched
        random-bases null), ``excess`` (mean - null_mean), ``n_groups``.
        All-nan dict if labels are unavailable or fewer than 2 usable groups.
    """
    nan_result = {
        "mean": float("nan"),
        "null_mean": float("nan"),
        "excess": float("nan"),
        "n_groups": 0,
    }
    labels = _extract_labels(dataset)
    if labels is None:
        warnings.warn(
            "SubspaceCommitment: no labels available, returning nan.", RuntimeWarning
        )
        return nan_result

    rng = np.random.default_rng(random_seed)
    bases_by_half, half_of_sample = group_half_bases(embeddings, labels, rank, rng)
    n_groups = len(bases_by_half[0])
    if n_groups < 2:
        warnings.warn(
            "SubspaceCommitment: fewer than 2 usable groups (>= 4 samples each), "
            "returning nan.",
            RuntimeWarning,
        )
        return nan_result

    # cross-fit: samples in half h are scored against bases fit on the OTHER half
    c = np.full(len(labels), np.nan)
    for h in (0, 1):
        mask = half_of_sample == h
        if mask.any():
            other = list(bases_by_half[1 - h].values())
            c[mask] = commitment_profile(embeddings[mask], other)
    mean = float(np.nanmean(c))

    d = embeddings.shape[1]
    ranks = [B.shape[1] for B in bases_by_half[0].values()]
    null_means = []
    for _ in range(n_null):
        null_bases = [np.linalg.qr(rng.standard_normal((d, r)))[0] for r in ranks]
        scored = embeddings[half_of_sample >= 0]
        null_means.append(float(np.nanmean(commitment_profile(scored, null_bases))))
    null_mean = float(np.nanmean(null_means)) if null_means else float("nan")

    return {
        "mean": mean,
        "null_mean": null_mean,
        "excess": mean - null_mean,
        "n_groups": n_groups,
    }
