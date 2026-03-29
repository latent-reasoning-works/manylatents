"""
Programmatic API for agent-driven workflows.

Hydra-free Python interface for manyLatents. Resolves string names via
Python registries, instantiates ``_target_`` dicts via importlib — no
GlobalHydra, no config composition, no singleton state.

The only Hydra fallback is for string metric bundle names (e.g.
``"standard"``) which require Hydra defaults composition.

Example:
    result = run(data='swissroll', algorithm='pca', metrics=['trustworthiness'])
    result = run(input_data=array, algorithm=PCAModule(n_components=5))
"""

from __future__ import annotations

import functools
import importlib
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instantiation helpers (no Hydra)
# ---------------------------------------------------------------------------


def _instantiate_target(cfg: dict, **extra) -> Any:
    """Instantiate a class from a dict with ``_target_``.

    Handles ``_partial_: True`` (returns functools.partial).
    Replaces ``hydra.utils.instantiate`` for the API path.
    """
    cfg = {**cfg, **extra}  # shallow merge, extra overrides cfg
    target = cfg.pop("_target_")
    partial = cfg.pop("_partial_", False)
    cfg.pop("_recursive_", None)  # Hydra meta-key, not needed
    cfg.pop("_convert_", None)

    module_path, class_name = target.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    if partial:
        return functools.partial(cls, **cfg)
    return cls(**cfg)


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _resolve_datamodule(input_data=None, data=None, seed=42, **kwargs):
    """Resolve data source to an instantiated LightningDataModule.

    Fast path: Python registry (no Hydra).
    Fallback: raises ValueError if not found.
    """
    if input_data is not None:
        from manylatents.data.precomputed_datamodule import PrecomputedDataModule

        logger.info(f"Wrapping input_data (shape={input_data.shape}) in PrecomputedDataModule")
        return PrecomputedDataModule(data=input_data, seed=seed, **kwargs)

    if data is not None:
        from manylatents.data import get_datamodule

        try:
            return get_datamodule(data, random_state=seed)
        except ValueError:
            raise
        except TypeError:
            # Constructor doesn't accept random_state — retry without it
            return get_datamodule(data)

    raise ValueError(
        "Either 'input_data' (np.ndarray) or 'data' (str dataset name) must be provided."
    )


def _resolve_algorithm(algorithm=None, algorithms=None, datamodule=None, seed=42, neighborhood_size=None, **kwargs):
    """Resolve algorithm specification to an instantiated module.

    Fast path: string name → Python registry.
    Dict with ``_target_`` → importlib instantiation (no Hydra).
    """
    from manylatents.algorithms.latent.latent_module_base import LatentModule

    # --- Pass-through: already-instantiated instance ---
    if algorithm is not None and not isinstance(algorithm, str):
        from lightning import LightningModule
        if isinstance(algorithm, (LatentModule, LightningModule)):
            return algorithm
        raise TypeError(
            f"algorithm must be a LatentModule, LightningModule, or string, "
            f"got {type(algorithm)}"
        )

    # --- String shorthand: algorithm="pca" ---
    if isinstance(algorithm, str):
        from manylatents.algorithms.latent import get_algorithm

        try:
            cls = get_algorithm(algorithm)
            algo_kwargs = {}
            if seed is not None:
                algo_kwargs["random_state"] = seed
            if neighborhood_size is not None:
                algo_kwargs["neighborhood_size"] = neighborhood_size
            return cls(**algo_kwargs)
        except KeyError:
            # Not in latent registry — try as algorithms dict
            algorithms = {"latent": algorithm}

    if algorithms is None:
        raise ValueError(
            "No algorithm specified. Provide 'algorithm' (instance or string) "
            "or 'algorithms' (dict)."
        )

    # --- Dict resolution ---
    algo_type = None
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
        # String in dict: {"latent": "pca"} — try registry
        if algo_type == "latent":
            from manylatents.algorithms.latent import get_algorithm
            try:
                cls = get_algorithm(algo_value)
                algo_kwargs = {}
                if seed is not None:
                    algo_kwargs["random_state"] = seed
                if neighborhood_size is not None:
                    algo_kwargs["neighborhood_size"] = neighborhood_size
                return cls(**algo_kwargs)
            except KeyError:
                pass

        raise ValueError(
            f"Algorithm '{algo_value}' not found in registry. "
            f"Use a dict with '_target_' for custom algorithms."
        )

    if isinstance(algo_value, dict):
        # Dict with _target_: instantiate via importlib
        algo_or_partial = _instantiate_target(algo_value, datamodule=datamodule)
        if isinstance(algo_or_partial, functools.partial):
            return algo_or_partial(datamodule=datamodule) if datamodule else algo_or_partial()
        return algo_or_partial

    raise TypeError(
        f"algorithms['{algo_type}'] must be a string or dict, "
        f"got {type(algo_value)}"
    )


def _resolve_metrics(metrics=None):
    """Resolve metrics specification.

    - ``None`` → no metrics
    - ``list[str]`` → registry names (passed through, no Hydra)
    - ``dict`` → configs with ``_target_`` (flattened + unrolled, no Hydra)
    - ``str`` → bundle name, requires Hydra compose (only Hydra touchpoint)

    Returns (engine_metrics, metrics_cfg) tuple.
    """
    if metrics is None:
        return None, None

    # list[str]: registry names — no Hydra
    if isinstance(metrics, list):
        return metrics, None

    # str: bundle name — this is the ONE path that needs Hydra
    if isinstance(metrics, str):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from pathlib import Path

        import manylatents.configs  # noqa: F401

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        config_dir = str((Path(__file__).parent / "configs").resolve())
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config", overrides=[f"metrics={metrics}"])

        if cfg.metrics is None:
            return None, None

        from manylatents.utils.metrics import flatten_and_unroll_metrics
        flattened = flatten_and_unroll_metrics(cfg.metrics)
        return flattened, cfg.metrics

    # dict: configs with _target_ — no Hydra
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

    Sampler instances (with ``get_indices``) pass through.
    Dicts with ``_target_`` are instantiated via importlib (no Hydra).
    """
    if sampling is None:
        return None

    result = {}
    for name, sampler_or_cfg in sampling.items():
        if hasattr(sampler_or_cfg, "get_indices"):
            result[name] = sampler_or_cfg
        elif isinstance(sampler_or_cfg, dict) and "_target_" in sampler_or_cfg:
            result[name] = _instantiate_target(sampler_or_cfg)
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
    Run a manyLatents experiment.

    Args:
        input_data: In-memory array (wraps in PrecomputedDataModule).
        data: Dataset name (e.g. ``"swissroll"``).
        algorithm: String name (``"pca"``), or pre-built instance.
        algorithms: Dict config, e.g. ``{"latent": "pca"}`` or
            ``{"latent": {"_target_": "...", "n_components": 2}}``.
        metrics: ``list[str]`` of registry names, ``dict`` of configs
            with ``_target_``, ``str`` bundle name, or ``None``.
        sampling: Dict of sampler configs or instances.
        seed: Random seed (default 42).
        **kwargs: ``neighborhood_size`` forwarded to algorithm.

    Returns:
        Dict with keys: embeddings, label, metadata, scores.

    Examples:
        >>> result = run(data='swissroll', algorithm='pca')
        >>> result = run(data='swissroll', algorithm='pca', metrics=['trustworthiness'])
        >>> result = run(input_data=array, algorithm=PCAModule(n_components=5))
    """
    from lightning import Trainer
    from manylatents.experiment import run_experiment

    neighborhood_size = kwargs.pop("neighborhood_size", None)

    datamodule = _resolve_datamodule(input_data=input_data, data=data, seed=seed)
    algo = _resolve_algorithm(
        algorithm=algorithm, algorithms=algorithms,
        datamodule=datamodule, seed=seed, neighborhood_size=neighborhood_size,
    )
    engine_metrics, metrics_cfg = _resolve_metrics(metrics)
    engine_sampling = _resolve_sampling(sampling)

    trainer = Trainer(
        accelerator="auto", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
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
