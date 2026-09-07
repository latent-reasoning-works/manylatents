"""Caller-controlled model-side dimension estimation from a fitted score model.

The score-vector SVD construction of Stanczuk et al. (ICML 2024) is legitimate:
near a manifold, scores at noisy probes span its normal space. This is a
score-vector construction, not a requirement to compute a score Jacobian.
The decision of how to read dimension from that evidence belongs to the method
owner. No estimator policy is selected by default.

The former largest-log-gap SELECTION RULE IS UNSOUND. Multiplying all scores
by a positive constant shifts their log singular values equally and leaves all
gaps unchanged (away from the numerical floor). At the origin, a point mass
and N(0, 25I) differ by precisely such a factor. In D=4, sigma=0.05, K=128,
seed=0, both returned 2 for one origin query and a mean of 1.625 for 16 repeated
origin queries. More probing cannot recover the discarded magnitude. 1.625 is
not universal: eight genuine Gaussian draws returned 3 at every point. Also,
with K >= D > 1 every cut is internal: only 1..D-1 can be returned, never 0 or D.
Clipping cannot restore these endpoints. The named ``largest_log_gap`` policy
therefore refuses; ``_largest_gap_cut`` retains the rule solely as a diagnostic.

FLIPD uses m_sigma(x) = D + sigma**2 * (div s_sigma(x) + ||s_sigma(x)||**2).
At the origin and sigma=0.05 this is 0 for the D=4 point mass and about 3.9996
for N(0, 25I). It retains magnitude information and permits endpoint dimensions;
a finite-scale value is not automatically a dimension estimate. The method
owner must supply BOTH scales and a rule interpreting behaviour across scales.
No noise-scale selection, threshold, plateau finder, or clipping is supplied.
"""
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Optional, Union

import numpy as np

from manylatents.metrics.registry import register_metric
from manylatents.utils.exceptions import MeasurementUnavailable


@dataclass
class ScoreDimensionEvidence:
    """Evidence passed to a caller's policy; all coordinates are standardized.

    ``singular_values`` lazily computes descending score-vector spectra of shape
    (N, min(K, D)). A policy returns one dimension per point, shape (N,), or raises
    MeasurementUnavailable. It can use ``module`` and ``points`` for other score
    constructions, as FLIPD does. ``sigma`` is explicit or module.sigma_min.
    """

    module: object
    points: np.ndarray
    sigma: float
    K: int
    seed: int

    @property
    def ambient_dim(self):
        return self.points.shape[1]

    @cached_property
    def singular_values(self):
        n, d = self.points.shape
        eps = np.random.default_rng(self.seed).standard_normal((n, self.K, d)).astype(np.float32)
        probes = (self.points[:, None, :] + self.sigma * eps).reshape(n * self.K, d)
        scores = np.asarray(self.module.score(probes, self.sigma, standardized=True))
        if scores.shape != probes.shape or not np.isfinite(scores).all():
            raise MeasurementUnavailable("Score spectrum requires finite scores of shape (N*K, D).")
        return np.linalg.svd(scores.reshape(n, self.K, d), compute_uv=False)


def _largest_gap_cut(sv, floor):
    """UNSOUND legacy diagnostic: number of SVs before the largest internal log gap.

    ``floor`` restricts candidate cuts relative to the largest SV; it does not
    repair scale blindness or allow endpoint ranks. Never a dimension policy.
    """
    sv = np.maximum(np.asarray(sv, float), 1e-12)
    if sv.size == 1:
        return 1
    gaps = np.diff(-np.log(sv))
    valid = (sv / sv[0])[:-1] >= floor
    if not valid.any():
        return 1
    return int(np.argmax(np.where(valid, gaps, -np.inf))) + 1


def largest_log_gap(evidence):
    """Named, unavailable legacy policy. See module docstring for measured failures."""
    raise MeasurementUnavailable(
        "largest_log_gap is unsound: scale-invariant gaps cannot distinguish a point mass "
        "from N(0, 25I) at the origin (both 2; repeated-origin mean 1.625), and internal "
        "cuts cannot express dimensions 0 or D. Supply a different estimator policy."
    )


def read_finite_scale(values, scales, *, sigma):
    """Read a caller-named finite-scale statistic, without dimension/limit inference.

    ``sigma`` is required and must occur exactly once in the supplied scales.
    This explicit column readout is useful for smoke fixtures; it does not test
    scale stability or choose scientifically appropriate noise scales.
    """
    indices = np.flatnonzero(np.asarray(scales) == sigma)
    if len(indices) != 1:
        raise MeasurementUnavailable("read_finite_scale requires sigma exactly once in scales.")
    return values[:, indices[0]]


@dataclass
class FLIPD:
    """FLIPD with caller-selected scales and interpretation, neither defaulted.

    ``scale_policy(values, scales)`` receives finite-scale m_sigma values of shape
    (N, S) and the supplied scales (S,), and returns dimensions (N,) or refuses.
    It owns the scientific decision about the scale limit/behaviour. For example,
    selecting a column only reports that finite-scale statistic, not a validated
    general dimension estimator.

    The module must implement ``score_tensor(z, sigma)``: differentiable torch
    scores in standardized coordinates for standardized input z. Divergence is
    exact autodiff for D <= exact_max_dim and a seeded Rademacher Hutchinson trace
    estimate otherwise (trace_samples probes). These are computational settings,
    not a policy for reading dimension. No estimates are clipped to [0, D].
    """

    scales: object
    scale_policy: Callable
    exact_max_dim: int = 32
    trace_samples: int = 128

    def __call__(self, evidence):
        import torch

        scales = np.asarray(self.scales, dtype=float)
        if scales.ndim != 1 or not scales.size or not np.isfinite(scales).all() or (scales <= 0).any():
            raise MeasurementUnavailable("FLIPD requires explicit finite positive noise scales.")
        if not callable(self.scale_policy):
            raise MeasurementUnavailable("FLIPD requires a caller-supplied scale_policy.")
        if self.exact_max_dim < 0 or self.trace_samples < 1:
            raise ValueError("FLIPD requires exact_max_dim >= 0 and trace_samples >= 1.")
        if not callable(getattr(evidence.module, "score_tensor", None)):
            raise MeasurementUnavailable("FLIPD requires a differentiable standardized score_tensor interface.")
        n, d = evidence.points.shape
        values = np.empty((n, scales.size), dtype=float)
        rng = np.random.default_rng(evidence.seed)
        with torch.enable_grad():
            for i, point in enumerate(evidence.points):
                for j, sigma in enumerate(scales):
                    z = torch.tensor(point[None, :], requires_grad=True)
                    score = evidence.module.score_tensor(z, float(sigma))
                    if not isinstance(score, torch.Tensor) or score.shape != z.shape or not score.requires_grad:
                        raise MeasurementUnavailable("FLIPD requires differentiable scores of shape (N, D).")
                    # Differentiate against the original input even if the module
                    # moves/casts it: score_tensor must preserve that graph.
                    if d <= self.exact_max_dim:
                        terms = []
                        for k in range(d):
                            grad = torch.autograd.grad(score[0, k], z, retain_graph=True, allow_unused=True)[0]
                            if grad is None:
                                raise MeasurementUnavailable("FLIPD score_tensor is disconnected from its input.")
                            terms.append(grad[0, k])
                        divergence = torch.stack(terms).sum()
                    else:
                        terms = []
                        for _ in range(self.trace_samples):
                            v = torch.as_tensor(rng.choice([-1., 1.], size=(1, d)), dtype=score.dtype, device=score.device)
                            grad = torch.autograd.grad((score * v).sum(), z, retain_graph=True, allow_unused=True)[0]
                            if grad is None:
                                raise MeasurementUnavailable("FLIPD score_tensor is disconnected from its input.")
                            terms.append((grad * v.to(grad.device)).sum())
                        divergence = torch.stack(terms).mean()
                    values[i, j] = d + sigma**2 * (float(divergence.detach()) + float(score.detach().square().sum()))
        if not np.isfinite(values).all():
            raise MeasurementUnavailable("FLIPD produced non-finite finite-scale statistics.")
        return self.scale_policy(values, scales.copy())


@register_metric(
    aliases=["score_id", "score_jacobian_id", "nb_dimension", "model_intrinsic_dim"],
    default_params={"return_per_sample": False},
    description="Model-side dimension shell requiring a caller-supplied estimator policy",
)
def ScoreJacobianID(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[object] = None,
    sigma: Optional[float] = None,
    K: Optional[int] = None,
    seed: int = 0,
    return_per_sample: bool = False,
    cache: Optional[dict] = None,
    estimator_policy=None,
) -> Union[float, np.ndarray]:
    """Evaluate a caller's policy on a cohort and return its mean or per-point values.

    Pass ``estimator_policy=callable`` taking ScoreDimensionEvidence and returning
    shape (N,). To read the spectrum use evidence.singular_values, ambient_dim,
    and sigma; the method owner supplies the significance/rank decision, including
    endpoint handling. To use FLIPD pass ``FLIPD(scales=..., scale_policy=...)``.
    Hydra can instantiate this policy via its _target_ and inject it here.
    Bare ``'flipd'`` refuses because scales and their interpretation are missing.
    ``'largest_log_gap'`` is named for discoverability but refuses as unsound.

    embeddings are raw (N, D) points, standardized by the fitted module. sigma
    defaults to module.sigma_min, K to max(2*D, 128); these control spectral probes,
    not FLIPD's explicit scales. Empty cohorts and missing policies refuse.
    Policy outputs must be finite (N,) values; no clipping or correction is made.
    """
    if estimator_policy is None:
        raise MeasurementUnavailable("ScoreJacobianID requires a caller-supplied estimator_policy; none was supplied.")
    if isinstance(estimator_policy, str):
        if estimator_policy == "largest_log_gap":
            return largest_log_gap(None)
        if estimator_policy == "flipd":
            raise MeasurementUnavailable("FLIPD requires FLIPD(scales=..., scale_policy=...); no noise-scale policy is supplied.")
        raise ValueError(f"Unknown estimator_policy: {estimator_policy!r}")
    if not callable(estimator_policy):
        raise TypeError("estimator_policy must be callable.")
    if module is None or not getattr(module, "_is_fitted", False):
        raise MeasurementUnavailable("ScoreJacobianID requires a fitted score module.")
    X = np.asarray(embeddings, np.float32)
    if X.ndim != 2 or not all(X.shape) or not np.isfinite(X).all():
        raise MeasurementUnavailable("ScoreJacobianID requires a nonempty finite cohort of shape (N, D).")
    sigma = getattr(module, "sigma_min", None) if sigma is None else sigma
    if sigma is None or not np.isfinite(sigma) or sigma <= 0:
        raise MeasurementUnavailable("ScoreJacobianID requires positive sigma or module.sigma_min.")
    if K is not None and (isinstance(K, bool) or int(K) != K or K < 1):
        raise ValueError("K must be a positive integer.")
    K = max(2 * X.shape[1], 128) if K is None else int(K)
    points = np.asarray(module._standardize(X), np.float32)
    if points.shape != X.shape or not np.isfinite(points).all():
        raise MeasurementUnavailable("ScoreJacobianID requires finite standardized points of shape (N, D).")
    evidence = ScoreDimensionEvidence(module, points, float(sigma), K, seed)
    result = np.asarray(estimator_policy(evidence), dtype=float)
    if result.shape != (len(X),) or not np.isfinite(result).all():
        raise MeasurementUnavailable("estimator_policy must return one finite dimension per cohort point, shape (N,).")
    return result if return_per_sample else float(result.mean())
