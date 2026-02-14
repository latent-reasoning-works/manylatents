"""Backend availability checks and resolution utilities.

Provides cached import guards for optional dependencies (TorchDR, FAISS)
and device/backend resolution helpers.
"""
import logging

import torch

logger = logging.getLogger(__name__)

_torchdr_available = None
_faiss_available = None


def check_torchdr_available() -> bool:
    """Check if TorchDR is importable. Result is cached after first call."""
    global _torchdr_available
    if _torchdr_available is None:
        try:
            import torchdr  # noqa: F401

            _torchdr_available = True
        except ImportError:
            _torchdr_available = False
    return _torchdr_available


def check_faiss_available() -> bool:
    """Check if FAISS is importable. Result is cached after first call."""
    global _faiss_available
    if _faiss_available is None:
        try:
            import faiss  # noqa: F401

            _faiss_available = True
        except ImportError:
            _faiss_available = False
    return _faiss_available


def resolve_device(device: str | None) -> str:
    """Resolve device string to concrete device.

    Args:
        device: None, "cpu", "cuda", or "auto".

    Returns:
        "cpu" or "cuda".
    """
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_backend(backend: str | None) -> str | None:
    """Resolve backend string.

    Args:
        backend: None, "sklearn", "torchdr", or "auto".

    Returns:
        None (use CPU library) or "torchdr".

    Raises:
        ImportError: If "torchdr" requested but not installed.
    """
    if backend is None or backend == "sklearn":
        return None
    if backend == "torchdr":
        if not check_torchdr_available():
            raise ImportError(
                "TorchDR backend requested but not installed. "
                "Install with: pip install manylatents[torchdr]"
            )
        return "torchdr"
    if backend == "auto":
        if check_torchdr_available() and torch.cuda.is_available():
            return "torchdr"
        return None
    raise ValueError(f"Unknown backend: {backend!r}. Use None, 'sklearn', 'torchdr', or 'auto'.")


def torchdr_knn_to_dense(model) -> 'torch.Tensor':
    """Convert TorchDR kNN-format affinity to dense NxN matrix.

    TorchDR stores affinities as (N, k) tensors alongside (N, k) neighbor
    indices. Some models also have a mask_affinity_in_ boolean mask indicating
    valid neighbors (UMAP uses this; TSNE does not).

    Args:
        model: A fitted TorchDR model with affinity_in_ and NN_indices_
               attributes.

    Returns:
        Dense (N, N) affinity tensor on the same device as the model output.
    """
    vals = model.affinity_in_.detach()
    nn_idx = model.NN_indices_.detach()
    N = vals.shape[0]

    # Build mask: use model's mask if available, otherwise all entries valid
    if hasattr(model, 'mask_affinity_in_') and model.mask_affinity_in_ is not None:
        mask = model.mask_affinity_in_.detach()
    else:
        mask = ~torch.isinf(vals)  # fallback: treat non-inf as valid

    # Zero out invalid entries
    vals_clean = vals.clone()
    vals_clean[torch.isinf(vals_clean)] = 0.0
    vals_clean[~mask] = 0.0

    # Build sparse NxN from kNN format
    row_indices = torch.arange(N, device=vals.device).unsqueeze(1).expand_as(nn_idx)
    row_flat = row_indices[mask].long()
    col_flat = nn_idx[mask].long()
    val_flat = vals_clean[mask]

    sparse_mat = torch.sparse_coo_tensor(
        torch.stack([row_flat, col_flat]),
        val_flat,
        size=(N, N),
    ).coalesce()

    dense = sparse_mat.to_dense()
    # Symmetrize: kNN graphs are directed (i→j doesn't imply j→i)
    return (dense + dense.T) / 2
