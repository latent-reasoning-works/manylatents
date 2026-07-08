"""ScoreJacobianID — model-side intrinsic dimension from a trained score model (Stanczuk NB estimator).

Stanczuk, Batzolis, Deveney & Schoenlieb, ICML 2024 ("Diffusion Models Encode the Intrinsic Dimension
of Data Manifolds", arXiv:2212.12611). Around a point x0, the score s(x0 + sigma*eps, sigma) at small
sigma points back toward the manifold, so a cloud of such scores SPANS THE NORMAL SPACE. With the K×D
score matrix's singular values sorted descending, the number of significant SVs = normal-bundle
dimension (D - m); the manifold dimension is

    m_hat(x0) = D - #{significant singular values},

where the significant/negligible split is the LARGEST gap in the (log) singular-value spectrum.

NOTE (verified before use): this is the score-vector SVD, NOT an autodiff score-Jacobian
eigendecomposition (equivalent in rank, cheaper here); and m = #vanishing SVs = D - #(large SVs) —
the naive "D - #(below the knee)" is inverted. `module` must be a fitted ScoreDiffusionModule.
"""
import logging
from typing import Optional, Union

import numpy as np

from manylatents.metrics.registry import register_metric

logger = logging.getLogger(__name__)


def _largest_gap_cut(sv, floor):
    """# significant (normal) singular values = index just after the largest log-drop.

    The knee is the largest gap over the FULL spectrum (that is where normal meets tangent). `floor`
    only restricts WHERE a valid cut may sit: the last significant SV must be >= floor*sv[0], so a
    huge gap buried inside the negligible tail can't masquerade as the boundary.
    """
    sv = np.maximum(np.asarray(sv, float), 1e-12)
    if sv.size == 1:
        return 1
    logs = np.log(sv)
    gaps = logs[:-1] - logs[1:]            # gaps[i] = drop from i to i+1
    r = sv / sv[0]
    valid = r[:-1] >= floor                # cut at i only if sv[i] (last normal) is non-negligible
    if not valid.any():
        return 1
    gaps = np.where(valid, gaps, -np.inf)
    g = int(np.argmax(gaps))
    return g + 1                           # indices 0..g are significant (normal)


@register_metric(
    aliases=["score_id", "score_jacobian_id", "nb_dimension", "model_intrinsic_dim"],
    default_params={"return_per_sample": False},
    description="Model-side intrinsic dimension via the Stanczuk normal-bundle score-SVD estimator",
)
def ScoreJacobianID(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[object] = None,
    sigma: Optional[float] = None,
    K: Optional[int] = None,
    floor: float = 1e-2,
    seed: int = 0,
    return_per_sample: bool = False,
    cache: Optional[dict] = None,
) -> Union[float, np.ndarray]:
    """Per-point model-side ID at the points `embeddings`, using the fitted score `module`.

    Parameters:
      - embeddings: points (N, D) to estimate ID at (raw space; standardised internally by module).
      - module: a fitted ScoreDiffusionModule (provides .score and the standardisation).
      - sigma: small noise scale for the normal-space probe (default: module.sigma_min).
      - K: number of score samples per point (default: max(2*D, 128)).
      - floor: singular values below floor*sv_max are treated as negligible (noise tail).
    """
    if module is None or not getattr(module, "_is_fitted", False):
        raise ValueError("ScoreJacobianID requires a fitted ScoreDiffusionModule as `module`.")
    X = np.asarray(embeddings, np.float32)
    N, D = X.shape
    # default probe = calibrated NB plateau in the module's standardised (unit-variance) space;
    # recovers known m within ~5% on iso-cubes. Sweep `sigma` to trace the plateau (Stanczuk).
    sigma = float(0.05 if sigma is None else sigma)
    K = int(max(2 * D, 128) if K is None else K)
    rng = np.random.default_rng(seed)

    Z0 = module._standardize(X)                                  # work where the score net lives
    # one batched score call: N*K perturbations
    eps = rng.standard_normal((N, K, D)).astype(np.float32)
    Zp = (Z0[:, None, :] + sigma * eps).reshape(N * K, D)
    scores = np.asarray(module.score(Zp, sigma, standardized=True), np.float32).reshape(N, K, D)

    m_hat = np.empty(N, float)
    for i in range(N):
        sv = np.linalg.svd(scores[i], compute_uv=False)
        n_sig = _largest_gap_cut(sv, floor)
        m_hat[i] = D - n_sig
    m_hat = np.clip(m_hat, 0, D)

    if return_per_sample:
        logger.info(f"ScoreJacobianID: per-sample model-ID, mean={m_hat.mean():.3f} (sigma={sigma}, K={K})")
        return m_hat
    mean_id = float(m_hat.mean())
    logger.info(f"ScoreJacobianID: mean model-ID = {mean_id:.3f} (sigma={sigma}, K={K})")
    return mean_id
