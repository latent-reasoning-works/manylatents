import logging
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from .latent_module_base import LatentModule, _to_numpy, _to_output

logger = logging.getLogger(__name__)


def _batched_loglog_r2(
    coords: torch.Tensor,
    neighbor_idx: torch.Tensor,
    k_values: torch.Tensor,
) -> torch.Tensor:
    """Differentiable batched R² of log(T_k) vs log(k).

    Args:
        coords: (n, 2) embedding coordinates.
        neighbor_idx: (n, max_k) neighbor indices (fixed).
        k_values: (n_k,) k values to evaluate at.

    Returns:
        (n,) R² per point.
    """
    # Distances to neighbors at each k value
    # neighbor_idx[:, k_values] gives the indices of the k-th neighbor
    nbr_idx = neighbor_idx[:, k_values]  # (n, n_k)
    nbr_coords = coords[nbr_idx]  # (n, n_k, 2)
    dists = (coords.unsqueeze(1) - nbr_coords).norm(dim=2)  # (n, n_k)

    log_d = torch.log(dists + 1e-30)
    log_k = torch.log(k_values.float() + 1.0)  # +1 because k_values are 0-indexed

    # Vectorized linear regression
    n_k = len(k_values)
    sum_x = log_k.sum()
    sum_x2 = (log_k ** 2).sum()
    sum_y = log_d.sum(dim=1)
    sum_xy = (log_d * log_k.unsqueeze(0)).sum(dim=1)

    denom = n_k * sum_x2 - sum_x ** 2
    slope = (n_k * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n_k

    y_pred = slope.unsqueeze(1) * log_k.unsqueeze(0) + intercept.unsqueeze(1)
    ss_res = ((log_d - y_pred) ** 2).sum(dim=1)
    y_mean = log_d.mean(dim=1, keepdim=True)
    ss_tot = ((log_d - y_mean) ** 2).sum(dim=1)

    r2 = torch.where(ss_tot > 1e-10, 1.0 - ss_res / ss_tot, torch.zeros_like(ss_tot))
    return r2.clamp(0.0, 1.0)


def _compute_mismatch_labels(
    input_data: np.ndarray,
    module: LatentModule,
    k_max: int = 200,
    k_min: int = 5,
    k_steps: int = 20,
    r2_threshold: float = 0.95,
    v_max: float = 1.0,
    v_min: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-point mismatch labels from log-log diagnostic + effective-k.

    Args:
        input_data: (n, d) high-dimensional input data.
        module: Fitted LatentModule with affinity_matrix().
        k_max: Maximum k for log-log diagnostic.
        k_min: Minimum k for log-log diagnostic.
        k_steps: Number of log-spaced k values.
        r2_threshold: R² threshold for valid geometric regime.
        v_max: Upper mismatch threshold (overshoot).
        v_min: Lower mismatch threshold (undershoot).

    Returns:
        mismatched: (n,) boolean mask.
        mismatch_ratio: (n,) v_i values.
    """
    from manylatents.utils.metrics import compute_knn

    # Step 1: log-log diagnostic on input space → k* per point
    cache = {}
    distances, _ = compute_knn(input_data, k=k_max, include_self=True, cache=cache)

    k_values = np.unique(
        np.logspace(np.log10(k_min), np.log10(k_max), k_steps).astype(int)
    )
    # Clamp to available columns (compute_knn clamps k when n_samples is small)
    max_col = distances.shape[1] - 1
    k_values = k_values[k_values <= max_col]

    T = distances[:, k_values]
    eps = 1e-30
    log_T = np.log(np.maximum(T, eps))
    log_k = np.log(k_values.astype(float))

    # Find k* per point: largest k where cumulative R² > threshold
    n_points = input_data.shape[0]
    k_star = np.full(n_points, k_values[-1], dtype=float)

    for i in range(n_points):
        for j in range(len(k_values), 2, -1):
            sub_log_T = log_T[i, :j]
            sub_log_k = log_k[:j]
            n_k = j
            sx = sub_log_k.sum()
            sx2 = (sub_log_k ** 2).sum()
            sy = sub_log_T.sum()
            sxy = (sub_log_T * sub_log_k).sum()
            d = n_k * sx2 - sx ** 2
            sl = (n_k * sxy - sx * sy) / d
            ic = (sy - sl * sx) / n_k
            yp = sl * sub_log_k + ic
            ss_res = ((sub_log_T - yp) ** 2).sum()
            ym = sub_log_T.mean()
            ss_tot = ((sub_log_T - ym) ** 2).sum()
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            if r2 >= r2_threshold:
                k_star[i] = k_values[j - 1]
                break
        else:
            k_star[i] = k_values[0]

    # Step 2: effective-k from affinity matrix
    try:
        W = module.affinity(ignore_diagonal=True, use_symmetric=False)
    except NotImplementedError:
        logger.warning(
            f"{module.__class__.__name__} has no affinity_matrix. "
            "Using uniform k_eff = neighborhood_size."
        )
        ns = getattr(module, 'neighborhood_size', None) or 15
        k_eff = np.full(n_points, float(ns))
    else:
        if hasattr(W, 'toarray'):
            W = W.toarray()
        W = np.asarray(W)
        row_sum = W.sum(axis=1)
        row_sum_sq = (W ** 2).sum(axis=1)
        k_eff = np.where(row_sum_sq > 0, row_sum ** 2 / row_sum_sq, 0.0)

    # Step 3: mismatch ratio
    mismatch_ratio = np.where(k_star > 0, k_eff / k_star, 0.0)
    mismatched = (mismatch_ratio > v_max) | (mismatch_ratio < v_min)

    logger.info(
        f"Mismatch labels: {mismatched.sum()}/{n_points} points flagged "
        f"(overshoot v>{v_max}: {(mismatch_ratio > v_max).sum()}, "
        f"undershoot v<{v_min}: {(mismatch_ratio < v_min).sum()})"
    )
    return mismatched, mismatch_ratio


class SelectiveCorrectionModule(LatentModule):
    """Wraps any LatentModule. Runs it, computes mismatch labels, corrects.

    The correction treats mismatched points' embedding coordinates as learnable
    parameters and optimizes them under a geometric consistency loss (log-log R²).
    Non-mismatched points are frozen. This is equivalent to an identity-initialized
    network with a gradient mask.

    Args:
        inner: The LatentModule to wrap (e.g., UMAPModule, TSNEModule).
        diagnostic_k: Maximum k for the log-log diagnostic on input space.
        r2_threshold: R² threshold defining the valid geometric regime.
        v_max: Upper mismatch threshold (overshoot). Default 1.0.
        v_min: Lower mismatch threshold (undershoot). Default 0.5.
        correction_lr: Learning rate for the correction optimizer.
        correction_steps: Number of optimization steps.
        correction_k: Number of neighbors for the embedding-space R² loss.
        correction_k_min: Minimum k for the embedding-space log-log sweep.
        correction_k_steps: Number of log-spaced k values for the sweep.
    """

    def __init__(
        self,
        inner: LatentModule,
        n_components: int = 2,
        random_state: int = 42,
        neighborhood_size: int | None = None,
        backend: str | None = None,
        device: str | None = None,
        diagnostic_k: int = 200,
        r2_threshold: float = 0.95,
        v_max: float = 1.0,
        v_min: float = 0.5,
        correction_lr: float = 1e-3,
        correction_steps: int = 500,
        correction_k: int = 100,
        correction_k_min: int = 5,
        correction_k_steps: int = 15,
        **kwargs,
    ):
        super().__init__(
            n_components=n_components,
            init_seed=random_state,
            backend=backend,
            device=device,
            neighborhood_size=neighborhood_size,
            **kwargs,
        )
        self.inner = inner
        self.diagnostic_k = diagnostic_k
        self.r2_threshold = r2_threshold
        self.v_max = v_max
        self.v_min = v_min
        self.correction_lr = correction_lr
        self.correction_steps = correction_steps
        self.correction_k = correction_k
        self.correction_k_min = correction_k_min
        self.correction_k_steps = correction_k_steps

        # Stored after fit
        self._input_data = None
        self._embedding = None
        self._mismatched = None
        self._mismatch_ratio = None
        self._correction_loss_history = None

    def fit(self, x, y=None) -> None:
        # Run inner algorithm
        self.inner.fit(x, y)
        self._input_data = _to_numpy(x)
        self._is_fitted = True

    def transform(self, x):
        if not self._is_fitted:
            raise RuntimeError("Not fitted. Call fit() first.")

        # Get inner embedding
        embedding = self.inner.transform(x)
        embedding_np = _to_numpy(embedding)

        # Compute mismatch labels
        self._mismatched, self._mismatch_ratio = _compute_mismatch_labels(
            self._input_data,
            self.inner,
            k_max=self.diagnostic_k,
            r2_threshold=self.r2_threshold,
            v_max=self.v_max,
            v_min=self.v_min,
        )

        n_mismatched = self._mismatched.sum()
        if n_mismatched == 0:
            logger.info("No mismatched points — returning inner embedding unchanged.")
            self._embedding = embedding_np
            return _to_output(embedding_np, x)

        logger.info(
            f"Correcting {n_mismatched}/{len(embedding_np)} points "
            f"({100 * n_mismatched / len(embedding_np):.1f}%)"
        )

        # Run SGD correction
        corrected = self._correct(embedding_np)
        self._embedding = corrected
        return _to_output(corrected, x)

    def _correct(self, embedding: np.ndarray) -> np.ndarray:
        """SGD correction of mismatched points."""
        from sklearn.neighbors import NearestNeighbors

        n = embedding.shape[0]
        mask = self._mismatched

        # Pre-compute fixed neighbor indices
        nn = NearestNeighbors(n_neighbors=self.correction_k)
        nn.fit(embedding)
        neighbor_idx = nn.kneighbors(return_distance=False)  # (n, k)

        # k grid for log-log sweep
        k_values = np.unique(
            np.logspace(
                np.log10(self.correction_k_min),
                np.log10(self.correction_k),
                self.correction_k_steps,
            ).astype(int)
        )
        # Ensure all within bounds
        k_values = k_values[k_values < self.correction_k]

        # Torch setup
        coords = torch.nn.Parameter(
            torch.tensor(embedding, dtype=torch.float32)
        )
        mask_t = torch.tensor(mask, dtype=torch.bool)
        nbr_idx_t = torch.tensor(neighbor_idx, dtype=torch.long)
        k_vals_t = torch.tensor(k_values, dtype=torch.long)

        optimizer = torch.optim.Adam([coords], lr=self.correction_lr)
        loss_history = []

        for step in range(self.correction_steps):
            optimizer.zero_grad()

            # Compute R² only for mismatched points
            r2_all = _batched_loglog_r2(coords, nbr_idx_t, k_vals_t)
            loss = (1.0 - r2_all[mask_t]).sum()

            loss.backward()

            # Zero gradients for non-mismatched points
            with torch.no_grad():
                coords.grad[~mask_t] = 0.0

            optimizer.step()
            loss_history.append(loss.item())

            if step % 100 == 0:
                logger.info(f"  Step {step}: loss={loss.item():.4f}")

        self._correction_loss_history = loss_history
        return coords.detach().numpy()

    def affinity(
        self, ignore_diagonal: bool = False, use_symmetric: bool = False
    ) -> np.ndarray:
        """Delegate to inner module."""
        return self.inner.affinity(
            ignore_diagonal=ignore_diagonal, use_symmetric=use_symmetric
        )

    def kernel(self, ignore_diagonal: bool = False) -> np.ndarray:
        """Delegate to inner module."""
        return self.inner.kernel(ignore_diagonal=ignore_diagonal)

    def adjacency(self, ignore_diagonal: bool = False) -> np.ndarray:
        """Delegate to inner module."""
        return self.inner.adjacency(ignore_diagonal=ignore_diagonal)

    def extra_outputs(self) -> dict:
        """Include mismatch diagnostics alongside inner module outputs."""
        extras = super().extra_outputs()
        if self._mismatched is not None:
            extras["mismatch_labels"] = self._mismatched
            extras["mismatch_ratio"] = self._mismatch_ratio
        if self._correction_loss_history is not None:
            extras["correction_loss_history"] = np.array(self._correction_loss_history)
        return extras
