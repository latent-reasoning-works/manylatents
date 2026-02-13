"""Tests for backend availability utilities."""
import pytest


def test_check_torchdr_available_returns_bool():
    """check_torchdr_available returns a boolean."""
    from manylatents.utils.backend import check_torchdr_available

    result = check_torchdr_available()
    assert isinstance(result, bool)


def test_check_torchdr_available_caches_result():
    """Subsequent calls use cached result without re-importing."""
    from manylatents.utils import backend

    # Reset cache
    backend._torchdr_available = None
    first = backend.check_torchdr_available()
    # Set to opposite to prove cache is used
    backend._torchdr_available = not first
    second = backend.check_torchdr_available()
    assert second == (not first)
    # Clean up
    backend._torchdr_available = None


def test_check_faiss_available_returns_bool():
    """check_faiss_available returns a boolean."""
    from manylatents.utils.backend import check_faiss_available

    result = check_faiss_available()
    assert isinstance(result, bool)


def test_resolve_device_returns_string():
    """resolve_device returns a valid device string."""
    from manylatents.utils.backend import resolve_device

    result = resolve_device(None)
    assert result in ("cpu", "cuda")

    result_cpu = resolve_device("cpu")
    assert result_cpu == "cpu"


def test_resolve_backend_none():
    """resolve_backend with None returns None."""
    from manylatents.utils.backend import resolve_backend

    assert resolve_backend(None) is None


def test_resolve_backend_sklearn():
    """resolve_backend with 'sklearn' returns None (CPU library)."""
    from manylatents.utils.backend import resolve_backend

    assert resolve_backend("sklearn") is None
