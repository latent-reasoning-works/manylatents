"""Unified metric evaluation API.

Provides both Hydra-free (registry-based) and Hydra (DictConfig-based)
metric evaluation through a single ``evaluate()`` entry point. Also
contains the cache pre-warming infrastructure (``extract_k_requirements``,
``prewarm_cache``) that was previously in experiment.py.

This module has NO imports from ``manylatents.experiment``.
"""
import copy as _copy
import inspect
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from manylatents.metrics.registry import get_metric
from manylatents.utils.metrics import _content_key, compute_eigenvalues, compute_knn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_metric_result(metric_name: str, raw_result: Any) -> dict[str, Any]:
    """Flatten a metric result into a dict of name->value pairs."""
    if isinstance(raw_result, dict):
        return {f"{metric_name}.{k}": v for k, v in raw_result.items()}
    return {metric_name: raw_result}


# ---------------------------------------------------------------------------
# Cache infrastructure
# ---------------------------------------------------------------------------


def extract_k_requirements(metrics) -> dict:
    """Scan metrics for kNN and spectral requirements, grouped by ``at`` value.

    Args:
        metrics: Either a Dict[str, DictConfig] (Hydra configs from
                 flatten_and_unroll_metrics) or a list[str] (metric registry
                 names from the programmatic API).

    Returns:
        Dict with keys:
            knn: dict mapping at_value -> set of k values needed
            spectral: whether eigendecomposition is needed
    """
    knn: dict[str, set[int]] = {}
    spectral = False

    if isinstance(metrics, list):
        # Registry path -- no ``at`` field, assume embedding
        for name in metrics:
            spec = get_metric(name)
            sig = inspect.signature(spec.func)
            k_val = None
            for param_name in ("k", "n_neighbors"):
                if param_name in spec.params:
                    val = spec.params[param_name]
                    if isinstance(val, (int, float)) and val > 0:
                        k_val = int(val)
                        break
                elif param_name in sig.parameters:
                    default = sig.parameters[param_name].default
                    if default is not inspect.Parameter.empty and isinstance(default, (int, float)) and default > 0:
                        k_val = int(default)
                        break

            if k_val is not None:
                knn.setdefault("embedding", set()).add(k_val)

    else:
        # DictConfig path -- use ``at`` field for routing
        for metric_name, metric_cfg in metrics.items():
            at_value = getattr(metric_cfg, "at", "embedding")

            # Module metrics need eigendecomposition
            if at_value == "module":
                spectral = True

            # Extract k value
            k_val = None
            for param in ("k", "n_neighbors"):
                if hasattr(metric_cfg, param):
                    val = getattr(metric_cfg, param)
                    if isinstance(val, (int, float)) and val > 0:
                        k_val = int(val)
                        break

            if k_val is not None:
                knn.setdefault(at_value, set()).add(k_val)

    return {"knn": knn, "spectral": spectral}


def prewarm_cache(
    metrics,  # Dict[str, DictConfig] or list[str]
    embeddings: np.ndarray,
    dataset,
    module=None,
    knn_cache_dir=None,
    cache: Optional[dict] = None,
    outputs: Optional[dict] = None,
) -> dict:
    """Pre-compute kNN and eigenvalues based on metric requirements.

    Uses the ``at`` field from each metric config to determine which data
    source needs kNN pre-computation.

    Args:
        metrics: Flattened metric configs (Dict[str, DictConfig]) or
            list of metric registry names (list[str]).
        embeddings: Embedding array.
        dataset: Dataset object with .data attribute.
        module: Optional fitted LatentModule.
        knn_cache_dir: Optional directory for disk-persisted dataset kNN.
        cache: Optional pre-existing cache dict. Created if None.
        outputs: Optional dict mapping at_value -> data arrays.
            Allows prewarming for any ``at`` value.

    Returns:
        Populated cache dict.
    """
    reqs = extract_k_requirements(metrics)
    if cache is None:
        cache = {}

    # kNN for each at_value that has requirements
    for at_value, k_set in reqs["knn"].items():
        data = None
        if outputs and at_value in outputs:
            data = outputs[at_value]
        elif at_value == "embedding":
            data = embeddings
        elif at_value == "dataset" and dataset is not None and hasattr(dataset, "data"):
            data = dataset.data

        if data is not None and k_set:
            max_k = max(k_set)

            # Disk cache for dataset kNN
            if at_value == "dataset" and knn_cache_dir is not None:
                content_hash = _content_key(data)
                npz_path = Path(knn_cache_dir) / "knn" / f"{content_hash}_k{max_k}.npz"
                if npz_path.exists():
                    saved = np.load(npz_path)
                    cache[content_hash] = (max_k, saved["distances"], saved["indices"])
                    logger.info(f"Loaded dataset kNN from disk cache: {npz_path}")
                    continue

            logger.info(f"Pre-warming cache: {at_value} kNN with max_k={max_k}")
            compute_knn(data, k=max_k, cache=cache)

            # Save dataset kNN to disk cache
            if at_value == "dataset" and knn_cache_dir is not None:
                content_hash = _content_key(data)
                knn_dir = Path(knn_cache_dir) / "knn"
                knn_dir.mkdir(parents=True, exist_ok=True)
                npz_path = knn_dir / f"{content_hash}_k{max_k}.npz"
                _, dists, idxs = cache[content_hash]
                np.savez(npz_path, distances=dists, indices=idxs)
                logger.info(f"Saved dataset kNN to disk cache: {npz_path}")

    # Eigenvalues if spectral metrics present
    if reqs["spectral"] and module is not None:
        logger.info("Pre-warming cache: eigenvalue decomposition")
        compute_eigenvalues(module, cache=cache)

    return cache


# ---------------------------------------------------------------------------
# Internal dispatch helpers
# ---------------------------------------------------------------------------


def _evaluate_registry(
    embeddings: np.ndarray,
    *,
    metrics: list[str],
    dataset=None,
    module=None,
    cache: dict,
    **kwargs,
) -> dict[str, Any]:
    """Evaluate metrics resolved from the registry by name.

    Args:
        embeddings: (n_samples, n_features) array.
        metrics: List of metric names or aliases from the registry.
        dataset: Optional dataset object.
        module: Optional fitted LatentModule.
        cache: Shared cache dict (already pre-warmed).
        **kwargs: Additional params forwarded to all metrics.

    Returns:
        Flat dict of metric results.
    """
    results: dict[str, Any] = {}
    for name in metrics:
        spec = get_metric(name)
        raw_result = spec(
            embeddings=embeddings,
            dataset=dataset,
            module=module,
            cache=cache,
            **kwargs,
        )
        results.update(_flatten_metric_result(name, raw_result))
    return results


def _evaluate_hydra(
    embeddings: np.ndarray,
    *,
    metrics: dict,
    outputs: dict[str, Any],
    dataset=None,
    module=None,
    cache: dict,
) -> dict[str, Any]:
    """Evaluate metrics from Hydra DictConfig with ``at`` field routing.

    Args:
        embeddings: (n_samples, n_features) array.
        metrics: Dict[str, DictConfig] from flatten_and_unroll_metrics.
        outputs: Dict mapping output names to data arrays.
        dataset: Optional dataset object.
        module: Optional fitted LatentModule.
        cache: Shared cache dict (already pre-warmed).

    Returns:
        Flat dict of metric results (with ColormapInfo viz metadata).
    """
    import hydra as _hydra
    from manylatents.callbacks.embedding.base import ColormapInfo

    results: dict[str, Any] = {}
    for metric_name, metric_cfg in metrics.items():
        # Read and pop 'at' before instantiation so it is not passed to the
        # metric function as a keyword argument.
        cfg_copy = _copy.deepcopy(metric_cfg)
        at_value = getattr(cfg_copy, "at", "embedding")  # default for safety
        if hasattr(cfg_copy, "at"):
            try:
                delattr(cfg_copy, "at")
            except (AttributeError, TypeError):
                pass  # read-only struct flags -- safe to ignore

        # Resolve primary data from outputs dict
        if at_value == "module":
            # Module metrics don't substitute the embeddings kwarg
            primary_data = None
        else:
            primary_data = outputs.get(at_value)
            if primary_data is None:
                algo_name = type(module).__name__ if module else "unknown"
                logger.warning(
                    f"Skipping metric '{metric_name}': output '{at_value}' "
                    f"not available from {algo_name}. "
                    f"Check that the algorithm produces this output."
                )
                continue

        metric_fn = _hydra.utils.instantiate(cfg_copy)

        # Call with standard signature -- route primary data to embeddings kwarg
        raw_result = metric_fn(
            embeddings=primary_data if primary_data is not None else embeddings,
            dataset=dataset,
            module=module,
            cache=cache,
        )

        # Unpack (value, ColormapInfo) tuples from metrics that provide viz metadata
        if (
            isinstance(raw_result, tuple)
            and len(raw_result) == 2
            and isinstance(raw_result[1], ColormapInfo)
        ):
            value, viz_info = raw_result
            results[metric_name] = value
            results[f"{metric_name}__viz"] = viz_info
        else:
            results.update(_flatten_metric_result(metric_name, raw_result))

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate(
    embeddings: np.ndarray,
    *,
    dataset=None,
    module=None,
    metrics=None,
    sampling: Optional[dict] = None,
    cache_dir: Optional[str] = None,
    cache: Optional[dict] = None,
    **kwargs,
) -> dict[str, Any]:
    """Unified metric evaluation entry point.

    Handles both ``list[str]`` (registry names from the programmatic API)
    and ``dict[str, DictConfig]`` (Hydra configs from
    ``flatten_and_unroll_metrics``).

    Args:
        embeddings: (n_samples, n_features) array or tensor.
        dataset: Dataset object with ``.data`` attribute.
        module: Optional fitted LatentModule.
        metrics: Either a list[str] of registry metric names or a
            dict[str, DictConfig] of Hydra metric configs.
        sampling: Optional dict mapping output names to already-instantiated
            sampler objects for post-fit sampling.
        cache_dir: Optional directory for disk-persisted kNN caches.
        cache: Optional shared cache dict. Created internally if None.
        **kwargs: Additional params forwarded to metrics (registry path only).

    Returns:
        Flat dict of metric results.
    """
    import torch

    if metrics is None:
        return {}

    # Ensure embeddings are numpy
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
        logger.debug(f"Converted embeddings from tensor to numpy: {embeddings.shape}")

    if cache is None:
        cache = {}

    # Build outputs dict for routing
    outputs: dict[str, Any] = {}
    if dataset is not None and hasattr(dataset, "data"):
        outputs["dataset"] = dataset.data
    outputs["embedding"] = embeddings
    if module is not None:
        outputs["module"] = module
        if hasattr(module, "extra_outputs"):
            for key, val in module.extra_outputs().items():
                outputs[key] = val

    # --- Post-fit sampling: dynamic over outputs dict ---
    embedding_sample_indices = None
    if sampling is not None:
        for output_name, sampler in sampling.items():
            if output_name == "dataset":
                continue  # pre-fit sampling handled upstream
            if output_name not in outputs or not isinstance(outputs[output_name], np.ndarray):
                continue
            indices = sampler.get_indices(outputs[output_name])
            outputs[output_name] = outputs[output_name][indices]
            logger.info(f"Post-fit sampling on '{output_name}': {len(indices)} samples using {type(sampler).__name__}")
            if output_name == "embedding":
                embedding_sample_indices = indices
        # If embedding was sampled, slice dataset to matching indices
        if embedding_sample_indices is not None and dataset is not None and hasattr(dataset, "data"):
            from manylatents.utils.sampling import _subsample_dataset_metadata
            dataset = _subsample_dataset_metadata(dataset, embedding_sample_indices)
            outputs["dataset"] = dataset.data
        # Sync local var with sampled output
        embeddings = outputs["embedding"]

    # Log dataset capabilities
    if dataset is not None:
        from manylatents.data.capabilities import log_capabilities
        log_capabilities(dataset)

    # --- Dispatch based on metrics type ---
    if isinstance(metrics, list):
        # Registry path (programmatic API)
        prewarm_cache(metrics, embeddings, dataset, module, cache=cache)
        return _evaluate_registry(
            embeddings,
            metrics=metrics,
            dataset=dataset,
            module=module,
            cache=cache,
            **kwargs,
        )
    else:
        # Hydra DictConfig path
        prewarm_cache(
            metrics, embeddings, dataset, module,
            knn_cache_dir=cache_dir, cache=cache, outputs=outputs,
        )
        return _evaluate_hydra(
            embeddings,
            metrics=metrics,
            outputs=outputs,
            dataset=dataset,
            module=module,
            cache=cache,
        )


def evaluate_metrics(
    embeddings: np.ndarray,
    *,
    metrics: list[str],
    module: Optional[object] = None,
    dataset: Optional[object] = None,
    cache: Optional[dict] = None,
    **kwargs,
) -> dict[str, Any]:
    """Evaluate named metrics on embeddings without Hydra.

    Backward-compatible alias for ``evaluate()`` with ``list[str]`` metrics.

    Args:
        embeddings: (n_samples, n_features) array.
        metrics: List of metric names or aliases from the registry.
        module: Optional fitted LatentModule (for module-context metrics).
        dataset: Optional dataset object (for dataset-context metrics).
        cache: Optional shared cache dict. Created internally if None.
            Pass a dict to share kNN/eigenvalue caches across calls.
        **kwargs: Additional params forwarded to all metrics.

    Returns:
        Flat dict of metric results. Dict-returning metrics are flattened
        with dotted keys (e.g., "k_eff.mean_k_eff").
    """
    return evaluate(
        embeddings,
        metrics=metrics,
        module=module,
        dataset=dataset,
        cache=cache,
        **kwargs,
    )
