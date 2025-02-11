import os

import numpy as np
from hydra.utils import to_absolute_path


def cache_result(cache_path: str, compute_func, *args, **kwargs):
    """
    Checks if a cached file exists. If so, load and return it.
    Otherwise, computes the result using compute_func, saves it, and returns it.

    Args:
        cache_path (str): Relative or absolute path for caching.
        compute_func (Callable): Function that computes the result.
        *args, **kwargs: Arguments passed to compute_func.

    Returns:
        The computed (or cached) result.
    """
    # Ensure the cache_path is absolute (using Hydra's utility, if needed)
    abs_cache_path = to_absolute_path(cache_path)
    os.makedirs(os.path.dirname(abs_cache_path), exist_ok=True)

    if os.path.exists(abs_cache_path):
        # Load the result; adjust this if your data isn’t NumPy
        return np.load(abs_cache_path, allow_pickle=True)
    else:
        result = compute_func(*args, **kwargs)
        np.save(abs_cache_path, result)
        return result
