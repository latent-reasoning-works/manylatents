"""Align a source trajectory onto a reference trajectory's step-grid.

A tier-2 :class:`LatentModule` (fit/transform), **not** a metric: it holds the
fitted step-correspondence as state and emits the source trajectory re-expressed
on the reference's step axis, so two reasoning trajectories — e.g. a *clean* vs a
*misleading-hint* run of the **same** model — become per-step comparable before
diffusion-geometry embedding (DiffusionMap/PHATE) or per-step divergence scoring.

Semantics (paired fit — the base's ``y`` slot carries the reference)::

    fit(source, reference)            # fit the step correspondence
    transform(source)   -> (T, d)     # source re-expressed on the reference grid
    fit_transform(source, reference)  # aligned source, ready for the diffusion embed
    residual(source, reference) -> (T,)  # per-step divergence curve

**v0 scope: same model.** Both trajectories share the hidden dim ``d``, so no
feature-space transform is needed (``mapper=None``). The ``mapper`` seam is
reserved for *cross-model* alignment (different ``d``), where a Procrustes/CCA
map is required — it raises ``NotImplementedError`` until that milestone. See the
design note for why cross-model (many manyagents LLMs → aligner) is the trigger.

Matchers (the step correspondence):

* ``"truncate"`` — compare the first ``min(S_source, S_reference)`` steps.
* ``"resample"`` — linearly interpolate the source onto the reference's step count
  (a normalized "% through reasoning" grid; useful for aggregating across problems).

If ``reference`` (``y``) is ``None`` the module self-aligns (identity): the source
is returned unchanged and the residual is zero — the degenerate reference-is-self
case and the identity bound the tests assert.
"""
from __future__ import annotations

import numpy as np

from manylatents.algorithms.latent.latent_module_base import (
    ArrayLike,
    LatentModule,
    _to_numpy,
    _to_output,
)

_MATCHERS = ("truncate", "resample")


class TrajectoryAligner(LatentModule):
    """Re-express a source trajectory on a reference trajectory's step-grid."""

    def __init__(self, matcher: str = "truncate", mapper: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if matcher not in _MATCHERS:
            raise ValueError(f"matcher must be one of {_MATCHERS}, got {matcher!r}")
        if mapper is not None:
            raise NotImplementedError(
                "feature-space mapper (cross-model alignment across different hidden "
                "dims) is not implemented in v0; same-model alignment uses mapper=None."
            )
        self.matcher = matcher
        self.mapper = mapper
        self._reference: np.ndarray | None = None

    def fit(self, x: ArrayLike, y: ArrayLike | None = None) -> None:
        source = _to_numpy(x)
        # y is the reference trajectory; None -> self-align (identity bound).
        reference = source if y is None else _to_numpy(y)
        if source.ndim != 2 or reference.ndim != 2:
            raise ValueError("source and reference must be 2-D (n_steps, d) trajectories")
        if source.shape[1] != reference.shape[1]:
            raise ValueError(
                f"same-model alignment requires equal hidden dim; got source "
                f"d={source.shape[1]} vs reference d={reference.shape[1]}. Cross-model "
                "(different d) needs the mapper seam, which is not implemented in v0."
            )
        self._reference = reference
        self._is_fitted = True

    def transform(self, x: ArrayLike) -> ArrayLike:
        if not self._is_fitted:
            raise RuntimeError("TrajectoryAligner is not fitted. Call `fit` first.")
        source, _ = self._align(_to_numpy(x), self._reference)
        return _to_output(source, x)

    def residual(self, x: ArrayLike, y: ArrayLike | None = None) -> np.ndarray:
        """Per-step divergence curve between the aligned source and reference.

        Cosine distance (``1 - cosine similarity``) per aligned step; shape ``(T,)``,
        non-negative, and exactly ``0`` where aligned steps coincide — so
        ``residual(A, A)`` is all zeros (the identity bound). Zero-norm steps are
        treated as aligned (distance 0). Pass ``y`` to score without fitting; else
        the reference stored by ``fit`` is used.
        """
        if y is None and not self._is_fitted:
            raise RuntimeError("TrajectoryAligner is not fitted; call `fit` first or pass `y`.")
        reference = self._reference if y is None else _to_numpy(y)
        src, ref = self._align(_to_numpy(x), reference)
        dot = np.sum(src * ref, axis=1)
        denom = np.linalg.norm(src, axis=1) * np.linalg.norm(ref, axis=1)
        cos = np.divide(dot, denom, out=np.ones_like(dot), where=denom > 0)
        return 1.0 - cos

    # -- matcher: (source, reference) -> (source_aligned, reference_aligned), both (T, d) --
    def _align(self, source: np.ndarray, reference: np.ndarray):
        if self.matcher == "truncate":
            t = min(source.shape[0], reference.shape[0])
            return source[:t], reference[:t]
        # "resample": put the source on the reference's step-grid
        return self._resample(source, reference.shape[0]), reference

    @staticmethod
    def _resample(traj: np.ndarray, n: int) -> np.ndarray:
        """Linearly interpolate ``traj`` (s, d) onto ``n`` evenly-spaced steps."""
        s = traj.shape[0]
        if s == n:
            return traj
        src_grid = np.linspace(0.0, 1.0, s)
        dst_grid = np.linspace(0.0, 1.0, n)
        return np.stack(
            [np.interp(dst_grid, src_grid, traj[:, j]) for j in range(traj.shape[1])],
            axis=1,
        )
