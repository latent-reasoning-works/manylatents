"""Centered Kernel Alignment (CKA) for representational similarity.

CKA is a global measure of similarity between two embedding spaces that is
invariant to orthogonal transformation and isotropic scaling.

Reference:
    Kornblith et al. "Similarity of Neural Network Representations Revisited"
    https://arxiv.org/abs/1905.00414
"""

from typing import Dict, Optional, Union
import numpy as np

from manylatents.metrics.registry import register_metric


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D, squeezing if needed."""
    if arr.ndim == 3 and arr.shape[1] == 1:
        return arr.squeeze(1)
    return arr


def _center_gram_matrix(K: np.ndarray) -> np.ndarray:
    """Center a Gram matrix (kernel matrix)."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Compute Hilbert-Schmidt Independence Criterion."""
    n = K.shape[0]
    return np.trace(K @ L) / (n - 1) ** 2


def _linear_kernel(X: np.ndarray) -> np.ndarray:
    """Compute linear kernel (Gram matrix)."""
    return X @ X.T


def _rbf_kernel(X: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    """Compute RBF (Gaussian) kernel."""
    sq_dists = (
        np.sum(X ** 2, axis=1, keepdims=True)
        + np.sum(X ** 2, axis=1, keepdims=True).T
        - 2 * X @ X.T
    )
    if sigma is None:
        sigma = np.sqrt(np.median(sq_dists[sq_dists > 0]) / 2)
        if sigma == 0:
            sigma = 1.0
    return np.exp(-sq_dists / (2 * sigma ** 2))


def cka_pairwise(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = "linear",
    sigma: Optional[float] = None,
) -> float:
    """Compute CKA between two representations.

    Args:
        X: First representation of shape (N, D1) or (N, 1, D1).
        Y: Second representation of shape (N, D2) or (N, 1, D2).
        kernel: Kernel type ("linear" or "rbf").
        sigma: RBF bandwidth (only used if kernel="rbf").

    Returns:
        CKA similarity in [0, 1].
    """
    X = _ensure_2d(X)
    Y = _ensure_2d(Y)

    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Sample count mismatch: {X.shape[0]} vs {Y.shape[0]}")

    if kernel == "linear":
        K = _linear_kernel(X)
        L = _linear_kernel(Y)
    elif kernel == "rbf":
        K = _rbf_kernel(X, sigma=sigma)
        L = _rbf_kernel(Y, sigma=sigma)
    else:
        raise ValueError(f"Unknown kernel: {kernel}. Use 'linear' or 'rbf'.")

    K_c = _center_gram_matrix(K)
    L_c = _center_gram_matrix(L)

    hsic_kl = _hsic(K_c, L_c)
    hsic_kk = _hsic(K_c, K_c)
    hsic_ll = _hsic(L_c, L_c)

    if hsic_kk == 0 or hsic_ll == 0:
        return 0.0

    return hsic_kl / np.sqrt(hsic_kk * hsic_ll)


@register_metric(
    aliases=["cka_linear"],
    default_params={"kernel": "linear"},
    description="Centered Kernel Alignment with linear kernel",
)
def CKA(
    embeddings: Union[np.ndarray, Dict[str, np.ndarray]],
    dataset=None,
    module=None,
    kernel: str = "linear",
    sigma: Optional[float] = None,
    return_matrix: bool = False,
    cache: Optional[dict] = None,
) -> Union[float, Dict[str, float], np.ndarray]:
    """Compute Centered Kernel Alignment across modalities.

    CKA measures global representational similarity, invariant to orthogonal
    transformation and isotropic scaling. Useful for comparing embedding
    spaces with different dimensions.

    Args:
        embeddings: Either a single array (N, D) or dict mapping modality
            names to arrays, e.g., {"esm3": (N, 1536), "evo2": (N, 1920)}.
        dataset: Optional dataset object (unused, for protocol).
        module: Optional module object (unused, for protocol).
        kernel: Kernel type ("linear" or "rbf").
        sigma: RBF bandwidth (only used if kernel="rbf").
        return_matrix: If True and embeddings is dict, return full CKA matrix.

    Returns:
        If embeddings is single array: 1.0 (self-similarity).
        If embeddings is dict: Dict mapping pair names to CKA values,
            e.g., {"esm3_evo2": 0.87}, or matrix if return_matrix=True.
    """
    # Single array case
    if isinstance(embeddings, np.ndarray):
        return 1.0  # Self-similarity

    # Multi-modal case
    modality_names = list(embeddings.keys())
    n_modalities = len(modality_names)

    if n_modalities < 2:
        raise ValueError("Need at least 2 modalities for CKA comparison")

    # Validate sample counts
    n_samples = _ensure_2d(embeddings[modality_names[0]]).shape[0]
    for name, emb in embeddings.items():
        if _ensure_2d(emb).shape[0] != n_samples:
            raise ValueError(f"Sample count mismatch: {name}")

    if return_matrix:
        cka_matrix = np.eye(n_modalities)
        for i in range(n_modalities):
            for j in range(i + 1, n_modalities):
                val = cka_pairwise(
                    embeddings[modality_names[i]],
                    embeddings[modality_names[j]],
                    kernel=kernel,
                    sigma=sigma,
                )
                cka_matrix[i, j] = val
                cka_matrix[j, i] = val
        return cka_matrix

    results = {}
    for i in range(n_modalities):
        for j in range(i + 1, n_modalities):
            pair_name = f"{modality_names[i]}_{modality_names[j]}"
            results[pair_name] = cka_pairwise(
                embeddings[modality_names[i]],
                embeddings[modality_names[j]],
                kernel=kernel,
                sigma=sigma,
            )
    return results
