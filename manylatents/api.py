"""
Programmatic API for agent-driven workflows.

This module provides a Python function interface for manyAgents to call
manyLatents directly without subprocess overhead. Instead of building a
full Hydra DictConfig and routing through a single Hydra config, the API
resolves string names to Python objects (DataModules, algorithms, metrics)
and delegates to :func:`~manylatents.experiment.run_experiment`.

Hydra is still used internally for *name resolution* (e.g. ``"swissroll"``
-> ``SwissRollDataModule``, ``"pca"`` -> ``PCAModule``), but the engine
itself operates on plain Python objects.

Example:
    # Single algorithm
    result = run(
        data='swissroll',
        algorithms={'latent': {'_target_': 'manylatents.algorithms.latent.pca.PCA', 'n_components': 10}}
    )

    # Chaining with input_data (multi-step via sequential API calls)
    result1 = run(data='swissroll', algorithms={'latent': 'pca'})
    result2 = run(input_data=result1['embeddings'], algorithms={'latent': 'phate'})
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hydra helpers (used only for string-name resolution)
# ---------------------------------------------------------------------------


def _hydra_compose(overrides: list[str]):
    """Compose a Hydra config with the given overrides.

    Manages GlobalHydra state so callers don't have to worry about
    leftover initializations from other libraries (manyagents, geomancy,
    shop, etc.).

    Returns an OmegaConf DictConfig.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # IMPORTANT: Import configs to register base_config with Hydra ConfigStore
    import manylatents.configs  # noqa: F401

    if GlobalHydra.instance().is_initialized():
        logger.debug("Clearing GlobalHydra before manylatents API call")
        GlobalHydra.instance().clear()

    config_dir = str((Path(__file__).parent / "configs").resolve())

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    return cfg


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _resolve_datamodule(input_data=None, data=None, seed=42, **kwargs):
    """Resolve data source to an instantiated LightningDataModule.

    Args:
        input_data: Optional np.ndarray of in-memory data.
        data: Optional string dataset name (e.g. ``"swissroll"``).
        seed: Random seed forwarded to the datamodule.
        **kwargs: Extra keyword arguments forwarded to PrecomputedDataModule
            when *input_data* is provided.

    Returns:
        A LightningDataModule instance (NOT yet ``.setup()``'d — the engine
        handles that).

    Raises:
        ValueError: If neither *input_data* nor *data* is provided.
    """
    if input_data is not None:
        from manylatents.data.precomputed_datamodule import PrecomputedDataModule

        logger.info(f"Wrapping input_data (shape={input_data.shape}) in PrecomputedDataModule")
        return PrecomputedDataModule(data=input_data, seed=seed, **kwargs)

    if data is not None:
        import hydra as _hydra
        from omegaconf import OmegaConf

        cfg = _hydra_compose([f"data={data}"])
        OmegaConf.set_struct(cfg, False)
        datamodule_cfg = {k: v for k, v in cfg.data.items() if k != "debug"}
        dm = _hydra.utils.instantiate(datamodule_cfg)
        return dm

    raise ValueError(
        "Either 'input_data' (np.ndarray) or 'data' (str dataset name) must be provided."
    )


def _resolve_algorithm(algorithm=None, algorithms=None, datamodule=None, seed=42, neighborhood_size=None, **kwargs):
    """Resolve algorithm specification to an instantiated module.

    Priority: *algorithm* (instance or string) > *algorithms* (dict).

    Args:
        algorithm: A pre-built LatentModule/LightningModule instance, or a
            string name (e.g. ``"pca"``).
        algorithms: A dict in the old-style format, e.g.
            ``{"latent": "pca"}`` or
            ``{"latent": {"_target_": "...", "n_components": 2}}``.
        datamodule: Optional datamodule passed to the algorithm constructor.
        seed: Random seed.
        neighborhood_size: Unified neighborhood parameter.
        **kwargs: Ignored (absorbs extra keyword arguments from ``run()``).

    Returns:
        An instantiated algorithm object.

    Raises:
        ValueError: If no algorithm can be resolved.
    """
    import hydra as _hydra
    from omegaconf import DictConfig, OmegaConf

    from manylatents.algorithms.latent.latent_module_base import LatentModule

    # --- Pass-through: already-instantiated instance ---
    if algorithm is not None and not isinstance(algorithm, str):
        # Check it's a LatentModule or LightningModule instance
        from lightning import LightningModule
        if isinstance(algorithm, (LatentModule, LightningModule)):
            logger.info(f"Using pre-built algorithm: {type(algorithm).__name__}")
            return algorithm
        raise TypeError(
            f"algorithm must be a LatentModule, LightningModule, or string, "
            f"got {type(algorithm)}"
        )

    # --- String shorthand: algorithm="pca" ---
    if isinstance(algorithm, str):
        algorithms = {"latent": algorithm}

    if algorithms is None:
        raise ValueError(
            "No algorithm specified. Provide 'algorithm' (instance or string) "
            "or 'algorithms' (dict)."
        )

    # --- Dict resolution via Hydra ---
    # Determine the config group and value
    algo_type = None  # "latent" or "lightning"
    algo_value = None

    for key in ("latent", "lightning"):
        if key in algorithms:
            algo_type = key
            algo_value = algorithms[key]
            break

    if algo_type is None:
        raise ValueError(
            f"algorithms dict must contain 'latent' or 'lightning' key, "
            f"got keys: {list(algorithms.keys())}"
        )

    if isinstance(algo_value, str):
        # String name -> Hydra compose
        overrides = [f"algorithms/{algo_type}={algo_value}"]
        if seed is not None:
            overrides.append(f"seed={seed}")
        if neighborhood_size is not None:
            overrides.append(f"neighborhood_size={neighborhood_size}")
        cfg = _hydra_compose(overrides)
        algo_cfg = getattr(cfg.algorithms, algo_type)
    elif isinstance(algo_value, dict):
        # Dict config -> convert to DictConfig
        algo_cfg = OmegaConf.create(algo_value)
    elif isinstance(algo_value, DictConfig):
        algo_cfg = algo_value
    else:
        raise TypeError(
            f"algorithms['{algo_type}'] must be a string or dict, "
            f"got {type(algo_value)}"
        )

    # Instantiate (handle _partial_ configs)
    algo_or_partial = _hydra.utils.instantiate(algo_cfg, datamodule=datamodule)
    if isinstance(algo_or_partial, functools.partial):
        if datamodule:
            return algo_or_partial(datamodule=datamodule)
        return algo_or_partial()
    return algo_or_partial


def _resolve_metrics(metrics=None):
    """Resolve metrics specification.

    Args:
        metrics: One of:
            - ``None`` -> no metrics
            - ``list[str]`` -> registry metric names (passed through)
            - ``dict`` -> Hydra-style metric configs (flattened + unrolled)
            - ``str`` -> bundle name (e.g. ``"standard"``) composed via Hydra

    Returns:
        ``(engine_metrics, metrics_cfg)`` tuple where:
        - *engine_metrics*: value suitable for ``run_experiment(metrics=...)``,
          either ``list[str]``, ``dict[str, DictConfig]``, or ``None``.
        - *metrics_cfg*: raw metric DictConfig for LightningModule model
          metrics, or ``None``.
    """
    if metrics is None:
        return None, None

    # --- list[str]: registry names, pass through ---
    if isinstance(metrics, list):
        return metrics, None

    # --- str: bundle name, compose via Hydra ---
    if isinstance(metrics, str):
        from omegaconf import OmegaConf

        cfg = _hydra_compose([f"metrics={metrics}"])
        if cfg.metrics is None:
            return None, None

        from manylatents.utils.metrics import flatten_and_unroll_metrics
        flattened = flatten_and_unroll_metrics(cfg.metrics)
        return flattened, cfg.metrics

    # --- dict or DictConfig: flatten and unroll ---
    if isinstance(metrics, dict):
        from omegaconf import DictConfig, OmegaConf
        from manylatents.utils.metrics import flatten_and_unroll_metrics

        if not isinstance(metrics, DictConfig):
            metrics_dc = OmegaConf.create(metrics)
        else:
            metrics_dc = metrics

        flattened = flatten_and_unroll_metrics(metrics_dc)
        return flattened, metrics_dc

    raise TypeError(
        f"metrics must be None, list[str], dict, or str, got {type(metrics)}"
    )


def _resolve_sampling(sampling=None):
    """Resolve sampling specification to instantiated sampler objects.

    Args:
        sampling: One of:
            - ``None`` -> no sampling
            - ``dict`` with sampler instances (objects with ``get_indices``)
              -> pass through
            - ``dict`` with Hydra-style sampler configs (containing
              ``_target_``) -> instantiate each

    Returns:
        Dict of instantiated sampler objects, or ``None``.
    """
    if sampling is None:
        return None

    import hydra as _hydra

    result = {}
    for name, sampler_or_cfg in sampling.items():
        # Already-instantiated sampler (has get_indices method)
        if hasattr(sampler_or_cfg, "get_indices"):
            result[name] = sampler_or_cfg
        elif isinstance(sampler_or_cfg, dict) and "_target_" in sampler_or_cfg:
            result[name] = _hydra.utils.instantiate(sampler_or_cfg)
        else:
            from omegaconf import DictConfig
            if isinstance(sampler_or_cfg, DictConfig) and "_target_" in sampler_or_cfg:
                result[name] = _hydra.utils.instantiate(sampler_or_cfg)
            else:
                raise TypeError(
                    f"sampling['{name}'] must be a sampler instance (with get_indices) "
                    f"or a dict with '_target_', got {type(sampler_or_cfg)}"
                )

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    input_data: np.ndarray | None = None,
    data: str | None = None,
    algorithm=None,
    algorithms: dict[str, Any] | None = None,
    metrics=None,
    sampling=None,
    seed: int = 42,
    **kwargs,
) -> dict[str, Any]:
    """
    Programmatic entry point for manyLatents.

    Resolves string names to Python objects and delegates to
    :func:`~manylatents.experiment.run_experiment`.

    Args:
        input_data: Optional input array. If provided, wraps in
            PrecomputedDataModule instead of loading from a configured dataset.
        data: Dataset name (e.g. ``"swissroll"``).  Mutually exclusive with
            *input_data*.
        algorithm: A pre-built LatentModule/LightningModule instance, or a
            string name (e.g. ``"pca"``).
        algorithms: Old-style dict, e.g. ``{"latent": "pca"}`` or
            ``{"latent": {"_target_": "...", "n_components": 2}}``.
        metrics: Metric specification — ``list[str]`` of registry names,
            ``dict`` of Hydra-style configs, ``str`` bundle name, or ``None``.
        sampling: Dict of sampler configs or pre-instantiated sampler objects.
        seed: Random seed (default 42).
        **kwargs: Extra keyword arguments. Recognized keys:
            - ``neighborhood_size``: Unified neighborhood parameter.
            All others are silently ignored.

    Returns:
        Dictionary with keys:
            - embeddings: The computed embeddings (numpy array)
            - label: Labels from the dataset (if available)
            - metadata: Dictionary with run metadata
            - scores: Evaluation metrics

    Examples:
        >>> # Single run
        >>> result = run(data='swissroll', algorithms={'latent': 'pca'})
        >>> embeddings = result['embeddings']
        >>>
        >>> # With full config
        >>> result = run(
        ...     data='swissroll',
        ...     algorithms={'latent': {'_target_': '...PCA', 'n_components': 10}},
        ...     metrics={'trustworthiness': {'_target_': '...', '_partial_': True, 'n_neighbors': 5, 'at': 'embedding'}},
        ... )
        >>>
        >>> # Chained runs
        >>> result1 = run(data='swissroll', algorithms={'latent': 'pca'})
        >>> result2 = run(input_data=result1['embeddings'], algorithms={'latent': 'phate'})
        >>>
        >>> # Pre-built algorithm
        >>> from manylatents.algorithms.latent.pca import PCAModule
        >>> pca = PCAModule(n_components=2)
        >>> result = run(data='swissroll', algorithm=pca)
    """
    from lightning import Trainer

    from manylatents.experiment import run_experiment

    neighborhood_size = kwargs.pop("neighborhood_size", None)

    # 1. Resolve data
    datamodule = _resolve_datamodule(
        input_data=input_data,
        data=data,
        seed=seed,
    )

    # 2. Resolve algorithm
    algo = _resolve_algorithm(
        algorithm=algorithm,
        algorithms=algorithms,
        datamodule=datamodule,
        seed=seed,
        neighborhood_size=neighborhood_size,
    )

    # 3. Resolve metrics
    engine_metrics, metrics_cfg = _resolve_metrics(metrics)

    # 4. Resolve sampling
    engine_sampling = _resolve_sampling(sampling)

    # 5. Create a minimal Trainer (API runs are always single-device, no logging)
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    # 6. Delegate to the Hydra-free engine
    logger.info(
        f"API run: algorithm={type(algo).__name__}, "
        f"data={'input_data' if input_data is not None else data}, "
        f"seed={seed}"
    )

    return run_experiment(
        datamodule=datamodule,
        algorithm=algo,
        trainer=trainer,
        metrics=engine_metrics,
        metrics_cfg=metrics_cfg,
        sampling=engine_sampling,
        seed=seed,
    )
