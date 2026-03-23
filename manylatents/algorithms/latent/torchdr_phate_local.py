"""Local TorchDR-style PHATE implementation for manylatents.

This module vendors the PHATE and PHATEAffinity logic used in TorchDR so
manylatents can run the same algorithm without importing torchdr.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
import torch


def _check_nans(x: torch.Tensor, msg: str) -> None:
    if torch.isnan(x).any() or torch.isinf(x).any():
        raise ValueError(msg)


def _pairwise_distances(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
    metric: str = "euclidean",
    k: Optional[int] = None,
    exclude_diag: bool = False,
    return_indices: bool = False,
):
    if Y is None:
        Y = X
        same = True
    else:
        same = Y is X

    if metric not in {"euclidean", "sqeuclidean"}:
        raise ValueError(f"Unsupported metric: {metric}")

    d = torch.cdist(X, Y, p=2)
    if metric == "sqeuclidean":
        d = d**2

    if same and exclude_diag:
        n = d.shape[0]
        idx = torch.arange(n, device=d.device)
        d[idx, idx] = float("inf")

    if k is not None:
        vals, inds = torch.topk(d, k=k, largest=False, dim=1)
        if return_indices:
            return vals, inds
        return vals

    if return_indices:
        return d, None
    return d


def _check_neighbor_param(k: int, n: int) -> int:
    if k <= 0:
        raise ValueError(f"k must be positive. Got {k}.")
    if k >= n:
        return max(1, n - 1)
    return k


class PHATEAffinity:
    def __init__(
        self,
        metric: str = "euclidean",
        device: str = "auto",
        backend: Optional[str] = None,
        verbose: bool = False,
        k: int = 5,
        alpha: float = 10.0,
        t: int = 5,
        thresh: Optional[float] = 1e-4,
        knn_max: Optional[int] = None,
        n_landmarks: Optional[int] = None,
        random_landmarking: bool = False,
        random_state: Optional[int] = None,
    ):
        if backend is not None:
            raise ValueError("Local PHATEAffinity currently supports backend=None only.")
        if metric != "euclidean":
            raise ValueError("Local PHATEAffinity currently supports metric='euclidean' only.")

        self.metric = metric
        self.device = device
        self.backend = backend
        self.verbose = verbose
        self.k = k
        self.alpha = alpha
        self.t = t
        self.thresh = thresh
        self.knn_max = knn_max
        self.n_landmarks = n_landmarks
        self.random_landmarking = random_landmarking
        self.random_state = random_state

        if self.thresh is not None and not (0 < self.thresh < 1):
            raise ValueError(f"thresh must be in (0, 1) or None. Got {self.thresh}.")
        if self.knn_max is not None and self.knn_max <= 0:
            raise ValueError(f"knn_max must be positive or None. Got {self.knn_max}.")
        if self.n_landmarks is not None and self.n_landmarks <= 1:
            raise ValueError(f"n_landmarks must be > 1 or None. Got {self.n_landmarks}.")

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self._compute_affinity(X)

    def _clear_landmark_state(self) -> None:
        for name in ["transitions_", "clusters_", "landmark_op_", "landmark_indices_"]:
            if hasattr(self, name):
                delattr(self, name)

    def _kmeans_assignments_torch(
        self, X: torch.Tensor, centers: torch.Tensor, chunk_size: int
    ) -> torch.Tensor:
        labels_chunks = []
        X = X.to(dtype=torch.float32)
        centers = centers.to(dtype=torch.float32)
        center_sq = (centers * centers).sum(dim=1)[None, :]

        for start in range(0, X.shape[0], chunk_size):
            x_chunk = X[start : start + chunk_size]
            x_sq = (x_chunk * x_chunk).sum(dim=1, keepdim=True)
            distances = x_sq + center_sq - 2.0 * (x_chunk @ centers.T)
            labels_chunks.append(torch.argmin(distances, dim=1))

        return torch.cat(labels_chunks, dim=0)

    def _kmeans_torch(
        self,
        X: torch.Tensor,
        n_clusters: int,
        max_iter: int = 30,
        tol: float = 1e-4,
    ) -> torch.Tensor:
        Xf = X.to(dtype=torch.float32)
        n = X.shape[0]
        if n_clusters >= n:
            return torch.arange(n, device=X.device, dtype=torch.long)

        generator = torch.Generator(device=X.device if X.device.type in {"cpu", "cuda"} else "cpu")
        if self.random_state is not None:
            generator.manual_seed(int(self.random_state))

        init_idx = torch.randperm(n, device=X.device, generator=generator)[:n_clusters]
        centers = Xf[init_idx].clone()

        target_entries = 16_000_000
        chunk_size = max(256, min(n, target_entries // max(1, n_clusters)))
        prev_labels = None

        for _ in range(max_iter):
            labels = self._kmeans_assignments_torch(Xf, centers, chunk_size=chunk_size)
            if prev_labels is not None and torch.equal(labels, prev_labels):
                break
            prev_labels = labels

            new_centers = torch.zeros_like(centers)
            new_centers.index_add_(0, labels, Xf)

            counts = torch.bincount(labels, minlength=n_clusters).to(dtype=torch.float32)
            non_empty = counts > 0
            new_centers[non_empty] = new_centers[non_empty] / counts[non_empty, None]

            empty = ~non_empty
            if bool(empty.any()):
                empty_idx = torch.where(empty)[0]
                refill_idx = torch.randint(
                    0,
                    n,
                    (empty_idx.shape[0],),
                    device=X.device,
                    generator=generator,
                )
                new_centers[empty_idx] = Xf[refill_idx]

            center_shift = torch.linalg.norm(new_centers - centers, dim=1).mean()
            centers = new_centers
            if center_shift <= tol:
                break

        return self._kmeans_assignments_torch(Xf, centers, chunk_size=chunk_size)

    def _build_landmark_operator(
        self, X: torch.Tensor, kernel: torch.Tensor, transition: torch.Tensor
    ) -> Optional[torch.Tensor]:
        n = X.shape[0]
        if self.n_landmarks is None or n <= self.n_landmarks:
            return None

        n_landmarks_target = int(self.n_landmarks)
        if n_landmarks_target <= 1:
            return None

        if self.random_landmarking:
            generator = torch.Generator(device=X.device if X.device.type in {"cpu", "cuda"} else "cpu")
            if self.random_state is not None:
                generator.manual_seed(int(self.random_state))
            permutation = torch.randperm(n, device=X.device, generator=generator)
            landmark_indices = permutation[:n_landmarks_target]

            distances = torch.cdist(
                X.to(torch.float32),
                X[landmark_indices].to(torch.float32),
            )
            cluster_assignments = distances.argmin(dim=1)
            self.landmark_indices_ = landmark_indices
        else:
            n_svd = min(100, max(2, n - 1))
            degrees = kernel.sum(dim=1, keepdim=True).clamp_min(1e-12)
            diff_aff = kernel / torch.sqrt(degrees @ degrees.T)
            _, _, V = torch.pca_lowrank(diff_aff.to(torch.float32), q=n_svd, center=False, niter=2)
            spectral_features = (transition @ V[:, :n_svd]).to(torch.float32)
            cluster_assignments = self._kmeans_torch(
                spectral_features, n_clusters=n_landmarks_target
            )

        _, cluster_assignments = torch.unique(
            cluster_assignments, sorted=True, return_inverse=True
        )
        n_landmarks_eff = int(cluster_assignments.max().item()) + 1

        pmn = torch.zeros((n_landmarks_eff, n), dtype=kernel.dtype, device=kernel.device)
        pmn.index_add_(0, cluster_assignments, kernel)

        pnm = pmn.T
        pmn = pmn / pmn.sum(dim=1, keepdim=True).clamp_min(1e-12)
        pnm = pnm / pnm.sum(dim=1, keepdim=True).clamp_min(1e-12)
        landmark_op = pmn @ pnm

        self.clusters_ = cluster_assignments
        self.transitions_ = pnm
        self.landmark_op_ = landmark_op
        return landmark_op

    def _compute_sparse_knn_kernel(self, X: torch.Tensor) -> torch.Tensor:
        n = X.shape[0]
        k_sigma = _check_neighbor_param(self.k, n)
        max_neighbors = n - 1
        if self.knn_max is not None:
            max_neighbors = min(int(self.knn_max), max_neighbors)
        if max_neighbors < k_sigma:
            raise ValueError(f"knn_max must be >= k. Got knn_max={self.knn_max}, k={self.k}.")

        if self.thresh is None:
            k_build = k_sigma
        else:
            k_build = max_neighbors

        knn_dist, knn_indices = _pairwise_distances(
            X,
            metric=self.metric,
            k=k_build,
            exclude_diag=True,
            return_indices=True,
        )

        sigma = knn_dist[:, k_sigma - 1].clamp_min(1e-12)
        weights = torch.exp(-((knn_dist / sigma[:, None]) ** self.alpha))

        keep_mask = torch.ones_like(weights, dtype=torch.bool)
        if self.thresh is not None:
            keep_mask = weights >= self.thresh
            keep_mask[:, :k_sigma] = True
            if (
                self.knn_max is not None
                and k_build < (n - 1)
                and bool((weights[:, -1] >= self.thresh).any())
            ):
                warnings.warn(
                    "PHATE thresholded alpha-decay neighborhoods hit the build cap "
                    f"(k_build={k_build}) while boundary affinity is still above thresh. "
                    "Increase knn_max for closer parity with graph-tools PHATE.",
                    RuntimeWarning,
                )

        self.sigma_ = sigma
        weights = weights * keep_mask.to(dtype=weights.dtype)

        kernel = torch.zeros((n, n), dtype=X.dtype, device=X.device)
        row_idx = torch.arange(n, device=X.device).unsqueeze(1).expand_as(knn_indices)
        kernel[row_idx, knn_indices.long()] = weights
        kernel.fill_diagonal_(1.0)
        return kernel

    def _compute_affinity(self, X: torch.Tensor) -> torch.Tensor:
        self._clear_landmark_state()

        kernel = self._compute_sparse_knn_kernel(X)
        kernel = (kernel + kernel.T) / 2
        transition = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-12)

        self.kernel_ = kernel
        self.transition_ = transition

        landmark_op = self._build_landmark_operator(X, kernel, transition)
        diffusion_source = landmark_op if landmark_op is not None else transition

        affinity = torch.linalg.matrix_power(diffusion_source.to(torch.float64), int(self.t))
        potential = -(affinity + 1e-7).log()
        affinity = -_pairwise_distances(potential, metric="euclidean")
        return affinity.to(X.dtype)


class PHATE:
    def __init__(
        self,
        k: int = 5,
        n_components: int = 2,
        t: int = 100,
        alpha: float = 10.0,
        decay: Optional[float] = None,
        knn_max: Optional[int] = None,
        thresh: Optional[float] = 1e-4,
        backend: Optional[str] = None,
        n_landmarks: Optional[int] = None,
        random_landmarking: bool = False,
        max_iter: int = 1000,
        init: str = "pca",
        init_scaling: float = 1.0,
        device: str = "auto",
        verbose: bool = False,
        random_state: Optional[int] = None,
        check_interval: int = 50,
        metric_in: str = "euclidean",
        mds_solver: str = "sgd",
        pairs_per_iter: Optional[int] = None,
        sgd_learning_rate: float = 1e-3,
        sgd_stress_tol: float = 1e-6,
    ):
        if backend is not None:
            raise ValueError("Local PHATE currently supports backend=None only.")

        if decay is not None:
            alpha = decay

        self.n_components = n_components
        self.metric_in = metric_in
        self.k = k
        self.t = t
        self.alpha = alpha
        self.knn_max = knn_max
        self.thresh = thresh
        self.n_landmarks = n_landmarks
        self.random_landmarking = random_landmarking
        self.max_iter = max_iter
        self.init = init
        self.init_scaling = init_scaling
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        self.check_interval = check_interval
        self.mds_solver = mds_solver
        self.pairs_per_iter = pairs_per_iter
        self.sgd_learning_rate = sgd_learning_rate
        self.sgd_stress_tol = sgd_stress_tol
        self.n_iter_ = -1

        if self.mds_solver != "sgd":
            raise ValueError(f"mds_solver must be 'sgd'. Got {self.mds_solver}.")
        if self.pairs_per_iter is not None and self.pairs_per_iter <= 0:
            raise ValueError(f"pairs_per_iter must be positive or None. Got {self.pairs_per_iter}.")
        if self.sgd_learning_rate <= 0:
            raise ValueError(f"sgd_learning_rate must be positive. Got {self.sgd_learning_rate}.")
        if self.sgd_stress_tol <= 0:
            raise ValueError(f"sgd_stress_tol must be positive. Got {self.sgd_stress_tol}.")

        self.affinity = PHATEAffinity(
            k=k,
            t=t,
            alpha=alpha,
            knn_max=knn_max,
            thresh=thresh,
            backend=backend,
            n_landmarks=n_landmarks,
            random_landmarking=random_landmarking,
            metric=metric_in,
            device=device,
            verbose=verbose,
            random_state=random_state,
        )

    def _to_tensor(self, X) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            Xt = X
        else:
            Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
        if Xt.dtype != torch.float32 and Xt.dtype != torch.float64:
            Xt = Xt.float()
        if self.device == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            dev = self.device
        return Xt.to(dev)

    def fit(self, X, y=None):
        self.embedding_ = self.fit_transform(X, y=y)
        return self

    def fit_transform(self, X, y=None):
        X = self._to_tensor(X)
        self._X_fit_shape = tuple(X.shape)
        self.embedding_ = self._fit_transform_sgd(X)
        return self.embedding_

    def transform(self, X):
        if not hasattr(self, "embedding_"):
            raise RuntimeError("PHATE model is not fitted yet. Call fit/fit_transform first.")
        X = self._to_tensor(X)
        if tuple(X.shape) == getattr(self, "_X_fit_shape", None):
            return self.embedding_.to(device=X.device, dtype=X.dtype)
        raise NotImplementedError(
            "Out-of-sample transform is not implemented in local PHATE yet. "
            "Call fit_transform on the target matrix."
        )

    def _fit_transform_sgd(self, X: torch.Tensor):
        if X.ndim != 2:
            raise ValueError(f"expected 2D input tensor, got shape={tuple(X.shape)}")

        affinity_matrix = self.affinity(X)
        self.affinity_in_ = affinity_matrix
        self.kernel_ = self.affinity.kernel_
        self.diff_op_ = self.affinity.transition_

        target_dist = (-self.affinity_in_).clamp_min(0.0)
        d_max = target_dist.max()
        if d_max > 0:
            target_dist = target_dist / d_max
        target_dist = target_dist.to(X.dtype)
        n = target_dist.shape[0]
        if n < 2:
            return target_dist.new_zeros((n, self.n_components))

        self._init_embedding_sgd(target_dist)

        if self.pairs_per_iter is None:
            pairs_per_iter = max(16, int(2 * n * math.log(max(n, 2))))
        else:
            pairs_per_iter = min(self.pairs_per_iter, n * n)

        total_pairs = max(n * (n - 1) / 2, 1.0)
        sampling_ratio = max(pairs_per_iter / total_pairs, 1e-12)
        eta_max = float(self.sgd_learning_rate) * math.sqrt(1.0 / sampling_ratio)
        eta_min = eta_max * 0.01
        decay = math.log(eta_max / eta_min) / max(self.max_iter - 1, 1)

        prev_stress = None
        for step in range(self.max_iter):
            self.n_iter_ = step
            lr_step = eta_max * math.exp(-decay * step)

            i = torch.randint(0, n, (pairs_per_iter,), device=self.embedding_.device)
            j = torch.randint(0, n, (pairs_per_iter,), device=self.embedding_.device)
            valid = i != j
            i = i[valid]
            j = j[valid]
            if i.numel() == 0:
                continue

            y_i = self.embedding_[i]
            y_j = self.embedding_[j]
            diff = y_i - y_j
            dist = diff.norm(dim=1).clamp_min(1e-10)
            target = target_dist[i, j]
            errors = target - dist

            weights = -2.0 * errors / dist
            grad_contrib = diff * weights[:, None]
            gradients = torch.zeros_like(self.embedding_)
            gradients.index_add_(0, i, grad_contrib)
            gradients.index_add_(0, j, -grad_contrib)

            with torch.no_grad():
                grad_norm = gradients.norm(2)
                if torch.isfinite(grad_norm) and grad_norm > 1e6:
                    gradients.mul_(1e6 / grad_norm)
                self.embedding_.sub_(lr_step * gradients)

            _check_nans(
                self.embedding_,
                msg=f"Local PHATE (sgd): NaNs in embedding at iter {step}.",
            )

            stress = (errors * errors).mean()
            if prev_stress is not None and step > 50:
                rel_change = torch.abs(stress - prev_stress) / (prev_stress + 1e-10)
                if rel_change.item() < self.sgd_stress_tol:
                    break
            prev_stress = stress

        if d_max > 0:
            with torch.no_grad():
                self.embedding_.mul_(d_max)

        transitions = getattr(self.affinity, "transitions_", None)
        if transitions is not None:
            transitions = transitions.to(device=self.embedding_.device, dtype=self.embedding_.dtype)
            self.landmark_embedding_ = self.embedding_
            self.embedding_ = transitions @ self.landmark_embedding_

        return self.embedding_

    def _init_embedding_sgd(self, target_dist: torch.Tensor):
        n = target_dist.shape[0]
        dtype = target_dist.dtype
        device = target_dist.device

        if isinstance(self.init, (torch.Tensor, np.ndarray)):
            embedding_ = torch.as_tensor(self.init, device=device, dtype=dtype)
        elif self.init in ("normal", "random"):
            embedding_ = torch.randn((n, self.n_components), device=device, dtype=dtype)
        elif self.init == "pca":
            try:
                D2 = target_dist.to(torch.float64) ** 2
                row_mean = D2.mean(dim=1, keepdim=True)
                col_mean = D2.mean(dim=0, keepdim=True)
                grand_mean = D2.mean()
                B = -0.5 * (D2 - row_mean - col_mean + grand_mean)

                q = min(max(self.n_components + 2, 4), max(2, n - 1))
                U, S, _ = torch.pca_lowrank(B, q=q, center=False, niter=2)
                vals = S[: self.n_components].clamp_min(0).sqrt()
                embedding_ = (U[:, : self.n_components] * vals.unsqueeze(0)).to(
                    device=device, dtype=dtype
                )
            except RuntimeError as err:
                warnings.warn(
                    f"Classical-MDS init failed ({err}). Falling back to random init.",
                    RuntimeWarning,
                )
                embedding_ = torch.randn((n, self.n_components), device=device, dtype=dtype)
        else:
            raise ValueError(f"init {self.init} not supported")

        with torch.no_grad():
            std = embedding_[:, 0].std().clamp_min(1e-12)
            embedding_ = float(self.init_scaling) * embedding_ / std

        self.embedding_ = embedding_.requires_grad_(True)
        return self.embedding_
