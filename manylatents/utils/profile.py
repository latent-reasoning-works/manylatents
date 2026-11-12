"""Deterministic materialisation and preparation for descriptive profiles."""
import numpy as np


def point_cloud_pca(points, *, max_components=50, center=True, scale=False,
                    dtype="float64", solver="full"):
    """Return centred, unscaled principal scores with deterministic signs."""
    if (max_components != 50 or center is not True or scale is not False
            or dtype != "float64" or solver != "full"):
        raise ValueError("pinned_profile_parameter")
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or 0 in x.shape or not np.isfinite(x).all():
        raise ValueError("invalid_point_cloud")
    x = x - x.mean(axis=0)
    u, singular, vt = np.linalg.svd(x, full_matrices=False)
    signs = np.sign(vt[np.arange(len(vt)), np.argmax(np.abs(vt), axis=1)])
    signs[signs == 0] = 1
    scores = u * singular * signs
    m = min(50, x.shape[1])
    result = np.zeros((len(x), m), dtype=np.float64)
    width = min(m, scores.shape[1])
    result[:, :width] = scores[:, :width]
    return result


def materialize_profile_input(data, *, seed=42, data_kwargs=None):
    """Load a named input once, before any recipe transformation."""
    from manylatents.api import _resolve_datamodule
    module = _resolve_datamodule(data=data, seed=seed, **(data_kwargs or {}))
    module.setup()
    dataset = module.train_dataset
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        return np.asarray(dataset.dataset.data)[dataset.indices]
    return np.asarray(dataset.data)
