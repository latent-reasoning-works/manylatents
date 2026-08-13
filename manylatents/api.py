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
import inspect
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


def _resolve_datamodule(input_data=None, data=None, seed=42, time=None, **kwargs):
    """Resolve data source to an instantiated LightningDataModule.

    Fast path: Python registry (no Hydra).
    Fallback: raises ValueError if not found.
    """
    if input_data is not None:
        from manylatents.data.precomputed_datamodule import PrecomputedDataModule

        logger.info(f"Wrapping input_data (shape={input_data.shape}) in PrecomputedDataModule")
        # ``time`` only applies to the in-memory path; named datasets carry their own.
        return PrecomputedDataModule(data=input_data, seed=seed, time=time, **kwargs)

    if data is not None:
        from manylatents.data import get_datamodule

        try:
            return get_datamodule(data, random_state=seed, **kwargs)
        except ValueError:
            raise
        except TypeError as e:
            # The retry exists for constructors that take no ``random_state``. Scope it to
            # exactly that: a TypeError naming any OTHER argument means a caller's
            # ``data_kwargs`` key is wrong, and swallowing it is how a parameterised dataset
            # silently becomes the default one — the caller then compares two configurations
            # that were never different.
            if "random_state" not in str(e):
                raise
            return get_datamodule(data, **kwargs)

    raise ValueError(
        "Either 'input_data' (np.ndarray) or 'data' (str dataset name) must be provided."
    )


#: Hydra meta-keys that are directives to the composer, not constructor arguments.
_META_KEYS = ("_target_", "_recursive_", "_convert_", "_partial_")


def _lightning_config(name: str):
    """The packaged config for a named lightning algorithm, or None if there isn't one.

    Located via ``importlib.resources`` rather than ``__file__`` so it survives being
    installed as a wheel. Interpolations are deliberately NOT resolved: every one of these
    configs carries ``datamodule: ${data}``, which only Hydra can fill, and which
    :func:`_instantiate_lightning` overrides with the real datamodule anyway.
    """
    from importlib import resources

    from omegaconf import OmegaConf

    path = resources.files("manylatents") / "configs" / "algorithms" / "lightning" / f"{name}.yaml"
    if not path.is_file():
        return None
    cfg = OmegaConf.load(str(path))
    # `default.yaml` is a composition stub with no `_target_` — not an algorithm.
    return cfg if "_target_" in cfg else None


def _target_accepts(target: str, key: str) -> bool:
    """Whether the class named by a ``_target_`` string takes ``key`` as an argument.

    True when it does, when it takes ``**kwargs``, and — deliberately — when we cannot tell:
    an unimportable target, a `_target_` with no dot in it, or a C-level signature is a
    failure of *our* introspection, and must not turn into a refusal of the caller's
    argument. The only thing this is allowed to do is reject a key that the class
    demonstrably does not have.
    """
    try:
        module_path, class_name = target.rsplit(".", 1)
        params = inspect.signature(
            getattr(importlib.import_module(module_path), class_name)
        ).parameters
    except (ImportError, AttributeError, ValueError, TypeError):
        return True
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return key in params


def _instantiate_lightning(cfg, datamodule, **overrides):
    """Build a LightningModule from a packaged config node.

    Nested `network` / `loss` / `optimizer` nodes are passed through AS CONFIGS, not
    instantiated: every one of these configs sets ``_recursive_: false`` because the modules
    instantiate their own sub-components in ``setup()``, once the input dimension is known
    from the data. Converting them to plain dicts here would break that — `Reconstruction`
    reads ``self.network_config.input_dim`` by attribute.

    ``overrides`` are the caller's ``**kwargs`` off :func:`run`, and used to be dropped on
    the floor: `run(algorithms={'lightning': 'mioflow'}, n_global_epochs=3)` trained
    mioflow.yaml's 100, and a misspelled parameter raised nothing at all — the same defect
    the two latent string forms were each fixed for (api.py:225, api.py:268).

    A nested dict PATCHES its node rather than replacing it, for two reasons: a bare dict
    would trade the silent drop for an AttributeError in `setup()` (reconstruction.py:45
    reads `network_config.input_dim` by attribute), and a wholesale replacement would
    discard every sibling key the packaged config exists to supply — `network={'latent_dim':
    3}` means "that one, keep the rest", not "here is a whole new network". A dict carrying a
    DIFFERENT ``_target_`` is a deliberate swap of the component, so that one replaces.

    A key inside a patch is checked against the node target's SIGNATURE, not against the keys
    the yaml happens to write, because those are not the same set: `cflows.yaml`'s optimizer
    node lists only `lr`, yet `torch.optim.Adam` takes `weight_decay`, and refusing that
    would make a routine override unreachable through the only form that patches. Checking
    here rather than letting `hydra_zen.instantiate` raise buys the error a full dataload
    earlier, in the caller's vocabulary rather than hydra's.
    """
    from omegaconf import DictConfig, OmegaConf

    target = str(cfg["_target_"])
    # Skip `datamodule` by KEY rather than reading and overriding it: OmegaConf resolves
    # interpolations on value access, so merely iterating `.items()` raises
    # InterpolationKeyError on `${data}`. We supply the real datamodule below regardless.
    kwargs = {k: cfg[k] for k in cfg.keys()
              if k not in _META_KEYS and k != "datamodule"}
    for key, value in overrides.items():
        node = kwargs.get(key)
        if isinstance(value, (dict, DictConfig)) and isinstance(node, DictConfig):
            node_target = node.get("_target_")
            if value.get("_target_", node_target) != node_target:
                kwargs[key] = OmegaConf.create(dict(value))
                continue
            for sub in value:
                if sub not in node and not _target_accepts(str(node_target), str(sub)):
                    raise TypeError(
                        f"{node_target} got unexpected keyword argument '{sub}' from the "
                        f"`{key}=` override. Check the spelling — a key that is in neither "
                        f"the packaged config nor the target's signature survives the merge "
                        f"and only fails later, inside setup()."
                    )
            kwargs[key] = OmegaConf.merge(node, value)
        elif isinstance(value, (dict, DictConfig)):
            kwargs[key] = OmegaConf.create(dict(value))
        else:
            kwargs[key] = value
    kwargs["datamodule"] = datamodule
    module_path, class_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**kwargs)


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
            # Remaining kwargs go to the algorithm constructor. They used to be dropped here:
            # only `random_state` and `neighborhood_size` were forwarded, so
            # `run(algorithm='pca', n_components=5)` silently returned a 2-column embedding,
            # and `diffusion_map` was unreachable by name at any dataset under 2000 rows
            # because its `n_landmark=2000` default could not be overridden.
            algo_kwargs = dict(kwargs)
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
                # Mirrors the `algorithm='<name>'` branch above. This was `{}`, so the two
                # string forms disagreed: `run(algorithm='pca', n_components=3)` honoured it
                # while `run(algorithms={'latent': 'pca'}, n_components=3)` silently returned
                # two columns — and the dict form is what geomancer's runner uses, so recipe
                # `params` never reached an algorithm through it.
                algo_kwargs = dict(kwargs)
                if seed is not None:
                    algo_kwargs["random_state"] = seed
                if neighborhood_size is not None:
                    algo_kwargs["neighborhood_size"] = neighborhood_size
                return cls(**algo_kwargs)
            except KeyError:
                pass

        # String in dict: {"lightning": "mioflow"} — resolve the packaged config by name.
        #
        # This branch did not exist: the registry lookup above was gated on `latent`, so a
        # string under `lightning` fell straight through to the raise below and EVERY
        # lightning algorithm was unreachable by name through the public API — mioflow,
        # cflows, latent_ode, aanet_reconstruction, ae_reconstruction alike. Only a
        # hand-written `_target_` dict worked, which meant a caller had to already know the
        # network, loss and optimizer wiring that the packaged configs exist to express.
        #
        # The unit resolved here is the CONFIG, not the class, because that is the useful
        # unit: `Reconstruction` is archetypal analysis only when its `network` is AAnet, and
        # `aanet_reconstruction.yaml` is what says so. Class-name lookup would have made
        # "reconstruction" ambiguous between AAnet and a plain autoencoder.
        if algo_type == "lightning":
            cfg = _lightning_config(algo_value)
            if cfg is not None:
                # Forwarding `kwargs` is what makes this string form agree with the two
                # latent string forms above, each of which fixed an earlier round of the same
                # bug. Without it the packaged yaml won every argument the caller named:
                # `run(algorithms={'lightning': 'mioflow'}, n_global_epochs=3)` trained 100,
                # and a misspelled parameter raised nothing at all, so a typo and a
                # deliberate default were the same observable.
                return _instantiate_lightning(cfg, datamodule, **kwargs)

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
    time: np.ndarray | None = None,
    data_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Run a manyLatents experiment.

    Args:
        input_data: In-memory array (wraps in PrecomputedDataModule).
        data: Dataset name (e.g. ``"swissroll"``).
        algorithm: String name (``"pca"``), or pre-built instance.
        algorithms: Dict config, e.g. ``{"latent": "pca"}``,
            ``{"lightning": "mioflow"}``, or
            ``{"latent": {"_target_": "...", "n_components": 2}}``.
        metrics: ``list[str]`` of registry names, ``dict`` of configs
            with ``_target_``, ``str`` bundle name, or ``None``.
        sampling: Dict of sampler configs or instances.
        seed: Random seed (default 42).
        time: Optional per-cell timepoint labels (in-memory path only), threaded to the
            datamodule so trajectory algorithms (LatentODE, Cflows) receive ``batch["time"]``.
            ``None`` (default) leaves every existing result unchanged.
        data_kwargs: Constructor kwargs for the named dataset's DataModule — the generation
            parameters of a synthetic dataset (``n_samples``, ``centers``, ``cluster_std``,
            ``n_branch``, ``concentration``, ``noise``, …). A separate channel from
            ``**kwargs`` on purpose: those go to the *algorithm*, and one bag for both would
            make ``n_components`` ambiguous. Unknown keys raise from the DataModule rather
            than being dropped.
        **kwargs: Constructor arguments for the ALGORITHM (``n_components``, ``knn``,
            ``n_landmark``, ``resolution``, …), on **every** string form —
            ``algorithm='pca'``, ``algorithms={'latent': 'pca'}`` and
            ``algorithms={'lightning': 'mioflow'}``. Previously nothing but
            ``neighborhood_size`` was forwarded on the latent forms, so
            ``run(algorithm='pca', n_components=5)`` quietly returned two columns; the
            lightning form forwarded nothing at all.

            A ``_target_`` dict is self-contained and does not consume these: put the
            arguments inside the dict instead.

            On the ``lightning`` key the arguments go to the LightningModule constructor
            (``n_global_epochs``, ``lambda_ot``, ``integration_times``, ``init_seed``, …). A
            nested node is PATCHED, not replaced: ``network={'latent_dim': 3}`` overrides
            that one key and keeps the packaged config's ``_target_`` and siblings. Pass a
            different ``_target_`` inside the node to swap the component wholesale.

            ``neighborhood_size`` is the one argument the lightning path still drops: it is a
            named parameter of ``run`` for the metric layer, no LightningModule accepts it,
            and forwarding it would break a mixed latent+lightning sweep that sets one value
            for both.

            Dataset generation parameters go through ``data_kwargs`` — the two are separate
            channels because ``n_components`` would otherwise be ambiguous between them.

    Returns:
        Dict with keys: embeddings, label, metadata, scores.

    Examples:
        >>> result = run(data='swissroll', algorithm='pca')
        >>> result = run(data='swissroll', algorithm='pca', metrics=['trustworthiness'])
        >>> result = run(input_data=array, algorithm=PCAModule(n_components=5))
        >>> result = run(data='gaussian_blob', algorithm='pca',
        ...              data_kwargs={'centers': 5, 'cluster_std': 0.4})
        >>> result = run(data='swissroll', algorithms={'lightning': 'mioflow'},
        ...              n_global_epochs=3, network={'hidden_dim': 32})
    """
    from lightning import Trainer
    from manylatents.experiment import run_experiment

    neighborhood_size = kwargs.pop("neighborhood_size", None)

    datamodule = _resolve_datamodule(input_data=input_data, data=data, seed=seed, time=time,
                                     **(data_kwargs or {}))
    algo = _resolve_algorithm(
        algorithm=algorithm, algorithms=algorithms,
        datamodule=datamodule, seed=seed, neighborhood_size=neighborhood_size,
        **kwargs,          # remaining kwargs are ALGORITHM constructor args; see below
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
