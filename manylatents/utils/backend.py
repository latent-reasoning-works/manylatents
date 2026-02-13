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
