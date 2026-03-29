import functools
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import lightning
import numpy as np
import torch
try:
    import wandb
    wandb.init  # verify real package, not wandb/ output dir
except (ImportError, AttributeError):
    wandb = None
from lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
)
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.callbacks.embedding.base import EmbeddingCallback, ColormapInfo
from manylatents.utils.data import determine_data_source
from manylatents.utils.metrics import _content_key, compute_knn, compute_eigenvalues, flatten_and_unroll_metrics
from manylatents.utils.utils import check_or_make_dirs, load_precomputed_embeddings, setup_logging, should_disable_wandb

logger = logging.getLogger(__name__)



def instantiate_datamodule(cfg: DictConfig, input_data_holder: Optional[Dict] = None) -> LightningDataModule:
    check_or_make_dirs(cfg.cache_dir)
    logger.info(f"Cache directory ensured at: {cfg.cache_dir}")
    datamodule_cfg = {k: v for k, v in cfg.data.items() if k != "debug"}

    # Inject input_data if provided (can't be in OmegaConf due to numpy array serialization)
    if input_data_holder is not None and 'data' in input_data_holder:
        datamodule_cfg['data'] = input_data_holder['data']

    dm = hydra.utils.instantiate(datamodule_cfg)
    return dm

def instantiate_algorithm(
    algorithm_config: DictConfig,
    datamodule: LightningDataModule | None = None,
) -> Any:
    """Instantiates the algorithm, handling partially configured objects."""
    algo_or_algo_partial = hydra.utils.instantiate(algorithm_config, datamodule=datamodule)
    if isinstance(algo_or_algo_partial, functools.partial):
        if datamodule:
            return algo_or_algo_partial(datamodule=datamodule)
        return algo_or_algo_partial()
    return algo_or_algo_partial

def instantiate_callbacks(
    trainer_cb_cfg: Dict[str, Any] = None,
    embedding_cb_cfg: Dict[str, Any] = None
) -> Tuple[List[Callback], List[EmbeddingCallback]]:
    trainer_cb_cfg   = trainer_cb_cfg   or {}
    embedding_cb_cfg = embedding_cb_cfg or {}

    lightning_cbs, embedding_cbs = [], []

    for name, one_cfg in trainer_cb_cfg.items():
        cb = hydra.utils.instantiate(one_cfg)
        if isinstance(cb, Callback):
            lightning_cbs.append(cb)
        else:
            logger.warning(f"Ignoring non‐Lightning callback '{name}'")

    for name, one_cfg in embedding_cb_cfg.items():
        cb = hydra.utils.instantiate(one_cfg)
        if isinstance(cb, EmbeddingCallback):
            embedding_cbs.append(cb)
        else:
            logger.warning(f"Ignoring non‐Embedding callback '{name}'")

    return lightning_cbs, embedding_cbs

def instantiate_trainer(
    cfg: DictConfig,
    lightning_callbacks: Optional[List] = None,
    loggers:            Optional[List] = None,
) -> Trainer:
    """
    Hydra-instantiate cfg.trainer by manually
    pulling out _target_, callbacks & logger fields.
    """
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)

    # remove hydra meta‐fields we don't want to forward
    ## hacky, needs to be fixed to conform with Trainer invocation
    trainer_kwargs.pop("_target_", None)
    trainer_kwargs.pop("callbacks", None)
    trainer_kwargs.pop("logger",    None)

    if lightning_callbacks:
        trainer_kwargs["callbacks"] = lightning_callbacks
    if loggers:
        trainer_kwargs["logger"] = loggers

    return Trainer(**trainer_kwargs)

@functools.singledispatch
def evaluate(algorithm: Any, /, **kwargs) -> Tuple[str, Optional[float], dict]:
    """Evaluates the algorithm.

    Returns the name of the 'error' metric for this run, its value, and a dict of metrics.
    """
    raise NotImplementedError(
        f"There is no registered handler for evaluating algorithm {algorithm} of type "
        f"{type(algorithm)}! (kwargs: {kwargs})"
    )

# --- Config sleuther ---


def extract_k_requirements(metrics) -> dict:
    """Scan metrics for kNN and spectral requirements, grouped by ``on`` value.

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
        import inspect
        from manylatents.metrics.registry import get_metric

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
        outputs: Optional dict mapping at_value -> data arrays, built by
            evaluate_outputs. Allows prewarming for any ``on`` value.

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
                from pathlib import Path
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
                from pathlib import Path
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


from manylatents.evaluate import _flatten_metric_result


@evaluate.register(dict)
def evaluate_outputs(
    latent_outputs: dict,
    *,
    cfg: DictConfig,
    datamodule,
    **kwargs,
) -> dict:
    import copy as _copy

    if latent_outputs is None or latent_outputs.get("embeddings") is None:
        logger.warning("No embeddings available for evaluation.")
        return {}

    embeddings = latent_outputs.get("embeddings")

    # Ensure embeddings are numpy arrays (metrics and sampling require numpy)
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
        logger.debug(f"Converted embeddings from tensor to numpy: {embeddings.shape}")

    # Handle different datamodule types - some store mode directly, others in hparams
    mode = getattr(datamodule, 'mode', None) or getattr(datamodule.hparams, 'mode', 'full')

    if mode == "split":
        ds = datamodule.test_dataset
    else:
        ds = datamodule.train_dataset  # defaults to full dataset on full runs

    logger.info(f"Reference data shape: {ds.data.shape}")

    module = kwargs.get("module", None)

    # Build outputs dict — maps output names to data
    outputs: dict[str, Any] = {}
    outputs["dataset"] = ds.data if hasattr(ds, "data") else None
    outputs["embedding"] = embeddings
    if module is not None:
        outputs["module"] = module
        for key, val in module.extra_outputs().items():
            outputs[key] = val

    # --- Post-fit sampling: dynamic over outputs dict ---
    sampling_cfg = OmegaConf.to_container(cfg.sampling, resolve=True) if hasattr(cfg, 'sampling') and cfg.sampling is not None else None
    embedding_sample_indices = None
    if sampling_cfg is not None:
        for output_name, sampler_cfg in sampling_cfg.items():
            if output_name == "dataset":
                continue  # pre-fit sampling handled in run_algorithm()
            if output_name not in outputs or not isinstance(outputs[output_name], np.ndarray):
                continue
            sampler = hydra.utils.instantiate(sampler_cfg)
            indices = sampler.get_indices(outputs[output_name])
            outputs[output_name] = outputs[output_name][indices]
            logger.info(f"Post-fit sampling on '{output_name}': {len(indices)} samples using {type(sampler).__name__}")
            if output_name == "embedding":
                embedding_sample_indices = indices
        # If embedding was sampled, slice dataset to matching indices for cross-space metrics
        if embedding_sample_indices is not None and hasattr(ds, 'data'):
            from manylatents.utils.sampling import _subsample_dataset_metadata
            ds = _subsample_dataset_metadata(ds, embedding_sample_indices)
            outputs["dataset"] = ds.data
        # Sync local vars with sampled outputs
        embeddings = outputs["embedding"]

    # Log dataset capabilities
    from manylatents.data.capabilities import log_capabilities
    log_capabilities(ds)

    metric_cfgs = flatten_and_unroll_metrics(cfg.metrics) if cfg.metrics is not None else {}

    # Pre-warm cache with optimal k values (uses sampled outputs)
    knn_cache_dir = getattr(cfg, "cache_dir", None)
    cache = prewarm_cache(
        metric_cfgs, embeddings, ds, module,
        knn_cache_dir=knn_cache_dir, outputs=outputs,
    )

    results: dict[str, Any] = {}
    for metric_name, metric_cfg in metric_cfgs.items():
        # Read and pop 'at' before instantiation so it is not passed to the
        # metric function as a keyword argument.
        cfg_copy = _copy.deepcopy(metric_cfg)
        at_value = getattr(cfg_copy, "at", "embedding")  # default for safety
        if hasattr(cfg_copy, "at"):
            try:
                delattr(cfg_copy, "at")
            except Exception:
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

        metric_fn = hydra.utils.instantiate(cfg_copy)

        # Call with standard signature -- route primary data to embeddings kwarg
        raw_result = metric_fn(
            embeddings=primary_data if primary_data is not None else embeddings,
            dataset=ds,
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

@evaluate.register(LightningModule)
def evaluate_lightningmodule(
    algorithm: LightningModule,
    *,
    cfg: DictConfig,
    trainer: Trainer,
    datamodule: Union[LightningDataModule, DataLoader],
    **kwargs,
) -> Tuple[dict, Optional[float]]:
    """
    Evaluate the LightningModule on the test set and compute additional custom metrics.
    
    Returns:
        A tuple: (combined_metrics, error_value).
        If evaluation is skipped (no test_step or empty results), returns ({}, None).
    """
    if not hasattr(algorithm, "test_step"):
        logger.info("Model does not define a test_step() method; skipping evaluation.")
        return {}, None

    results = trainer.test(model=algorithm, datamodule=datamodule)
    if not results:
        return {}, None

    base_metrics = results[0]
    custom_metrics = {}
    model_metrics_cfg = cfg.metrics.get("model", {}) if cfg.metrics is not None else {}

    for metric_key, metric_params in model_metrics_cfg.items():
        if metric_params.get("enabled", True):
            metric_fn = hydra.utils.instantiate(metric_params)
            # Let any errors during metric computation propagate.
            name, value = metric_fn(algorithm, test_results=base_metrics)
            custom_metrics[name] = value

    combined_metrics = {**base_metrics, **custom_metrics}
    error_value = next(iter(combined_metrics.values())) if combined_metrics else None
    return combined_metrics, error_value


def execute_step(
    algorithm: Any,
    train_tensor: torch.Tensor,
    test_tensor: torch.Tensor,
    trainer: lightning.Trainer,
    cfg: DictConfig,
    datamodule: Optional[LightningDataModule] = None,
    train_labels: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """
    Core execution engine for a single algorithm step.

    Args:
        algorithm: The algorithm instance (LatentModule or LightningModule)
        train_tensor: Training data tensor
        test_tensor: Test data tensor
        trainer: The Lightning trainer instance
        cfg: The Hydra configuration
        datamodule: Optional datamodule for LightningModule training/evaluation
        train_labels: Optional labels for supervised LatentModules (e.g., ClassifierModule)

    Returns:
        The computed latent embeddings as a tensor
    """
    latents = None

    # --- Compute latents based on algorithm type ---
    if isinstance(algorithm, LatentModule):
        # LatentModule: fit/transform pattern
        # Pass labels for supervised modules (ignored by unsupervised)
        algorithm.fit(train_tensor, train_labels)
        try:
            latents = algorithm.transform(test_tensor)
        except NotImplementedError:
            # Transductive backends (e.g. TorchDR) don't support out-of-sample
            # transform — fall back to fit_transform on test data directly
            logger.warning(
                f"{type(algorithm).__name__} does not support transform(). "
                "Falling back to fit_transform() on test data (transductive mode)."
            )
            latents = algorithm.fit_transform(test_tensor)
        logger.info(f"LatentModule embedding shape: {latents.shape}")

    elif isinstance(algorithm, LightningModule):
        # LightningModule: training with optional pretrained checkpoint

        # Load pretrained checkpoint if specified (skip training)
        if hasattr(cfg, 'pretrained_ckpt') and cfg.pretrained_ckpt:
            logger.info(f"Loading pretrained model from {cfg.pretrained_ckpt}")
            algorithm = LightningModule.load_from_checkpoint(cfg.pretrained_ckpt)
        else:
            # Training phase
            logger.info("Running training...")
            trainer.fit(algorithm, datamodule=datamodule)

        # Model evaluation
        logger.info("Running model evaluation.")
        evaluation_result = evaluate(
            algorithm,
            cfg=cfg,
            trainer=trainer,
            datamodule=datamodule,
        )

        # Handle evaluation results
        if isinstance(evaluation_result, tuple):
            model_metrics, model_error = evaluation_result
        else:
            model_metrics, model_error = evaluation_result, None

        logger.info(f"Model evaluation completed. Error: {model_error}, Metrics: {model_metrics}")

        # Extract embeddings from encoder
        if hasattr(algorithm, "encode"):
            logger.info("Extracting embeddings using network encoder...")
            latents = algorithm.encode(test_tensor)
            latents = latents.detach().cpu().numpy() if isinstance(latents, torch.Tensor) else latents
            logger.info(f"LightningModule embedding shape: {latents.shape}")
        else:
            logger.warning(f"LightningModule {type(algorithm).__name__} has no 'encode' method - skipping")

    return latents


def run_algorithm(cfg: DictConfig, input_data_holder: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Execute a single algorithm experiment.

    This is the core experiment logic that:
      - Instantiates datamodule, algorithms, and trainer
      - Computes embeddings through a unified latent module interface
      - Runs training and evaluation

    Args:
        cfg: The Hydra configuration

    Returns:
        A dictionary with keys: embeddings, label, metadata, scores
    """
    setup_logging(debug=cfg.debug, log_level=getattr(cfg, "log_level", "warning"))
    logger.info("Final Config:\n" + OmegaConf.to_yaml(cfg))

    # Initialize wandb if logger config is provided and not disabled
    wandb_run = None
    wandb_disabled = should_disable_wandb(cfg) or wandb is None

    if not wandb_disabled and cfg.logger is not None:
        logger.info(f"Initializing wandb logger: {OmegaConf.to_yaml(cfg.logger)}")
        # Call wandb.init directly to avoid hydra.instantiate config parameter conflict
        wandb_run = wandb.init(
            project=cfg.logger.get("project", cfg.project),
            name=cfg.logger.get("name", cfg.name),
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.logger.get("mode", "online"),
            dir=os.environ.get("WANDB_DIR", "logs"),
        )
    else:
        logger.info("WandB logging disabled - skipping wandb initialization")

    lightning.seed_everything(cfg.seed, workers=True)

    # --- Data instantiation ---
    datamodule = instantiate_datamodule(cfg, input_data_holder)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()
    field_index, data_source = determine_data_source(train_loader)

    # --- Algorithm module ---
    # Determine which algorithm to instantiate based on configuration
    if hasattr(cfg.algorithms, 'latent') and cfg.algorithms.latent is not None:
        algorithm = instantiate_algorithm(cfg.algorithms.latent, datamodule)
    elif hasattr(cfg.algorithms, 'lightning') and cfg.algorithms.lightning is not None:
        algorithm = instantiate_algorithm(cfg.algorithms.lightning, datamodule)
    else:
        raise ValueError("No algorithm specified in configuration")

    # --- Callbacks ---
    # Merge callbacks from both cfg.trainer.callbacks and cfg.callbacks.trainer
    trainer_cb_cfg = dict(cfg.trainer.get("callbacks") or {})
    if hasattr(cfg, "callbacks") and cfg.callbacks is not None:
        # Also check cfg.callbacks.trainer for Lightning callbacks (e.g., probe callback)
        extra_trainer_cbs = cfg.callbacks.get("trainer") or {}
        trainer_cb_cfg.update(extra_trainer_cbs)
    embedding_cb_cfg = cfg.callbacks.get("embedding") if hasattr(cfg, "callbacks") else None

    lightning_cbs, embedding_cbs = instantiate_callbacks(
        trainer_cb_cfg,
        embedding_cb_cfg
    )
    if lightning_cbs:
        logger.info(f"Instantiated {len(lightning_cbs)} Lightning callbacks: {[type(cb).__name__ for cb in lightning_cbs]}")

    if not embedding_cbs:
        logger.info("No embedding callbacks configured; skip embedding‐level hooks.")

    # --- Loggers ---
    # Only instantiate trainer loggers if top-level logger is enabled
    # This keeps LatentModule and LightningModule logging in sync
    # Use the centralized should_disable_wandb() check for consistency
    loggers = []
    if not wandb_disabled and cfg.logger is not None:
        for lg_conf in cfg.trainer.get("logger", {}).values():
            loggers.append(hydra.utils.instantiate(lg_conf))
        logger.info(f"Trainer loggers enabled: {len(loggers)} logger(s)")
    else:
        logger.info("Trainer loggers disabled (WandB disabled)")

    # --- Trainer ---
    trainer = instantiate_trainer(
        cfg,
        lightning_callbacks=lightning_cbs,
        loggers=loggers,
    )

    logger.info("Starting algorithm execution...")

    # --- Algorithm Execution ---
    embeddings: Dict[str, Any] = {}

    if OmegaConf.select(cfg, "eval_only", default=False):
        logger.info("Evaluation-only mode: Loading precomputed latent outputs.")
        embeddings = load_precomputed_embeddings(cfg)
    else:
        # Unroll train and test dataloaders to obtain tensors
        train_tensor = torch.cat([b[field_index].cpu() for b in train_loader], dim=0)
        test_tensor  = torch.cat([b[field_index].cpu() for b in test_loader],  dim=0)

        # Extract labels for supervised LatentModules (if available)
        train_labels = None
        train_dataset = getattr(datamodule, "train_dataset", None)
        if train_dataset is not None and hasattr(train_dataset, "get_labels"):
            labels = train_dataset.get_labels()
            if labels is not None:
                train_labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
                logger.info(f"Extracted {len(train_labels)} training labels for supervised learning")

        # --- Pre-fit sampling (optional) ---
        sampling_cfg = OmegaConf.to_container(cfg.sampling, resolve=True) if hasattr(cfg, 'sampling') and cfg.sampling is not None else None
        pre_fit_indices = None
        if sampling_cfg is not None and "dataset" in sampling_cfg:
            dataset_sampler = hydra.utils.instantiate(sampling_cfg["dataset"])
            pre_fit_indices = dataset_sampler.get_indices(
                train_tensor.numpy() if torch.is_tensor(train_tensor) else train_tensor
            )
            train_tensor = train_tensor[pre_fit_indices]
            test_tensor = test_tensor[pre_fit_indices]
            if train_labels is not None:
                train_labels = train_labels[pre_fit_indices]
            logger.info(f"Pre-fit sampling: {len(pre_fit_indices)} samples using {type(dataset_sampler).__name__}")

        logger.info(
            f"Running algorithm on {data_source}:\n"
            f"Train tensor shape: {train_tensor.shape}\n"
            f"Test tensor shape: {test_tensor.shape}\n"
            f"Algorithm: {type(algorithm).__name__}"
        )

        # Execute the algorithm step
        t_total_start = time.perf_counter()
        t_step_start = time.perf_counter()
        latents = execute_step(
            algorithm=algorithm,
            train_tensor=train_tensor,
            test_tensor=test_tensor,
            trainer=trainer,
            cfg=cfg,
            datamodule=datamodule,
            train_labels=train_labels,
        )
        step_time = time.perf_counter() - t_step_start

        # --- Unified embedding wrapping ---
        if latents is not None:
            embeddings = {
                "embeddings": latents,
                "label": getattr(getattr(datamodule, "test_dataset", None), "get_labels", lambda: None)(),
                "metadata": {
                    "source": "single_algorithm",
                    "algorithm_type": type(algorithm).__name__,
                    "data_shape": test_tensor.shape,
                    "step_time": step_time,
                },
            }

            # Attach extra outputs (affinity_matrix, adjacency_matrix, etc.)
            if isinstance(algorithm, LatentModule):
                extras = algorithm.extra_outputs()
                for key, val in extras.items():
                    embeddings[key] = val
                    shape_info = f" shape={val.shape}" if hasattr(val, 'shape') else ""
                    logger.info(f"Extra output attached: {key}{shape_info}")

            # Evaluate embeddings
            logger.info(f"Evaluating embeddings from {type(algorithm).__name__}...")
            t_eval_start = time.perf_counter()
            embeddings["scores"] = evaluate(
                embeddings,
                cfg=cfg,
                datamodule=datamodule,
                module=algorithm if isinstance(algorithm, LatentModule) else None
            )
            eval_time = time.perf_counter() - t_eval_start
            total_time = time.perf_counter() - t_total_start

            embeddings["metadata"]["eval_time"] = eval_time
            embeddings["metadata"]["total_time"] = total_time

    # --- Callback processing ---
    callback_outputs = {}
    if embeddings and embedding_cbs:
        for cb in embedding_cbs:
            cb_result = cb.on_latent_end(dataset=datamodule.test_dataset, embeddings=embeddings)
            # Merge callback outputs into the main callback_outputs dict
            if isinstance(cb_result, dict):
                callback_outputs.update(cb_result)
                logger.info(f"Callback {cb.__class__.__name__} returned: {list(cb_result.keys())}")

    # Add callback outputs to embeddings dict if any were generated
    if callback_outputs:
        embeddings['callback_outputs'] = callback_outputs
        logger.info(f"Added callback outputs to embeddings: {list(callback_outputs.keys())}")

    logger.info("Experiment complete.")

    # Auto-log scores to wandb if active (works online and offline)
    if wandb_run is not None and embeddings.get("scores"):
        scores = embeddings["scores"]
        scalar_metrics = {}
        for name, val in scores.items():
            if isinstance(val, tuple) and len(val) == 2:
                scalar_metrics[f"metrics/{name}"] = float(val[0])
            elif np.ndim(val) == 0:
                scalar_metrics[f"metrics/{name}"] = float(val)
        if scalar_metrics:
            wandb.log(scalar_metrics)
            logger.info(f"Auto-logged {len(scalar_metrics)} metrics to wandb: {list(scalar_metrics.keys())}")

    # Clean up wandb run if it was initialized
    if wandb_run is not None:
        wandb.finish()

    return embeddings