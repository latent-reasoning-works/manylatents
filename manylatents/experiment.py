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
from manylatents.utils.utils import check_or_make_dirs, load_precomputed_embeddings, setup_logging

logger = logging.getLogger(__name__)


def should_disable_wandb(cfg: DictConfig) -> bool:
    """
    Determine if WandB should be disabled based on configuration.

    WandB is disabled when:
    1. logger is explicitly set to None (orchestrated by parent like Geomancer)
    2. debug mode is True (fast testing/CI)
    3. WANDB_MODE environment variable is set to 'disabled'

    This ensures consistent behavior across run_algorithm() and run_pipeline().

    Args:
        cfg: Hydra configuration

    Returns:
        True if WandB should be disabled, False otherwise
    """
    # Check explicit logger=None (orchestrated mode)
    if cfg.logger is None:
        logger.info("WandB disabled: logger=None (orchestrated by parent)")
        return True

    # Check debug mode
    if cfg.debug:
        logger.info("WandB disabled: debug=True")
        return True

    # Check environment variable (allows external override)
    if os.environ.get('WANDB_MODE', '').lower() == 'disabled':
        logger.info("WandB disabled: WANDB_MODE=disabled")
        return True

    return False


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

# Metrics known to need dataset-space kNN (not just embedding-space)
_DATA_KNN_METRICS = frozenset({
    "manylatents.metrics.knn_preservation.KNNPreservation",
    "manylatents.metrics.trustworthiness.Trustworthiness",
    "manylatents.metrics.continuity.Continuity",
})

# Metrics known to need eigendecomposition of module affinity matrix
_SPECTRAL_METRICS = frozenset({
    "manylatents.metrics.spectral_gap_ratio.SpectralGapRatio",
    "manylatents.metrics.spectral_decay_rate.SpectralDecayRate",
    "manylatents.metrics.affinity_spectrum.AffinitySpectrum",
    "manylatents.metrics.dataset_topology_descriptor.DatasetTopologyDescriptor",
})


def extract_k_requirements(metric_cfgs: Dict[str, DictConfig]) -> dict:
    """Scan metric configs for kNN and spectral requirements.

    Args:
        metric_cfgs: Flattened metric configs from flatten_and_unroll_metrics().

    Returns:
        Dict with keys:
            emb_k: set of k values needed on embeddings
            data_k: set of k values needed on original data
            spectral: whether eigendecomposition is needed
    """
    emb_k: set[int] = set()
    data_k: set[int] = set()
    spectral = False

    for metric_cfg in metric_cfgs.values():
        target = getattr(metric_cfg, "_target_", "")

        # Extract k value
        k_val = None
        for param in ("k", "n_neighbors"):
            if hasattr(metric_cfg, param):
                val = getattr(metric_cfg, param)
                if isinstance(val, (int, float)) and val > 0:
                    k_val = int(val)
                    break

        if k_val is not None:
            emb_k.add(k_val)
            if target in _DATA_KNN_METRICS:
                data_k.add(k_val)

        if target in _SPECTRAL_METRICS:
            spectral = True

    return {"emb_k": emb_k, "data_k": data_k, "spectral": spectral}


def prewarm_cache(
    metric_cfgs: Dict[str, DictConfig],
    embeddings: np.ndarray,
    dataset,
    module=None,
    knn_cache_dir=None,
) -> dict:
    """Pre-compute kNN and eigenvalues based on metric requirements.

    Args:
        metric_cfgs: Flattened metric configs.
        embeddings: Embedding array.
        dataset: Dataset object with .data attribute.
        module: Optional fitted LatentModule.
        knn_cache_dir: Optional directory for disk-persisted dataset kNN.

    Returns:
        Populated cache dict.
    """
    reqs = extract_k_requirements(metric_cfgs)
    cache: dict = {}

    if reqs["emb_k"]:
        max_k = max(reqs["emb_k"])
        logger.info(f"Pre-warming cache: embedding kNN with max_k={max_k}")
        compute_knn(embeddings, k=max_k, cache=cache)

    if reqs["data_k"] and dataset is not None and hasattr(dataset, "data"):
        max_k = max(reqs["data_k"])
        content_hash = _content_key(dataset.data)

        # Try loading from disk cache
        loaded = False
        if knn_cache_dir is not None:
            from pathlib import Path
            npz_path = Path(knn_cache_dir) / "knn" / f"{content_hash}_k{max_k}.npz"
            if npz_path.exists():
                saved = np.load(npz_path)
                cache[content_hash] = (max_k, saved["distances"], saved["indices"])
                logger.info(f"Loaded dataset kNN from disk cache: {npz_path}")
                loaded = True

        if not loaded:
            logger.info(f"Pre-warming cache: dataset kNN with max_k={max_k}")
            compute_knn(dataset.data, k=max_k, cache=cache)

            # Save to disk cache
            if knn_cache_dir is not None:
                from pathlib import Path
                knn_dir = Path(knn_cache_dir) / "knn"
                knn_dir.mkdir(parents=True, exist_ok=True)
                npz_path = knn_dir / f"{content_hash}_k{max_k}.npz"
                _, dists, idxs = cache[content_hash]
                np.savez(npz_path, distances=dists, indices=idxs)
                logger.info(f"Saved dataset kNN to disk cache: {npz_path}")

    if reqs["spectral"] and module is not None:
        logger.info("Pre-warming cache: eigenvalue decomposition")
        compute_eigenvalues(module, cache=cache)

    return cache


@evaluate.register(dict)
def evaluate_embeddings(
    latent_outputs: dict,
    *,
    cfg: DictConfig,
    datamodule,
    **kwargs,
) -> dict:
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

    # Subsample for large datasets using pluggable sampling strategies
    ds_sub, emb_sub = ds, embeddings
    if cfg.metrics is not None:
        sampling_cfg = cfg.metrics.get("sampling", None)
        if sampling_cfg is not None:
            sampler = hydra.utils.instantiate(sampling_cfg)
            emb_sub, ds_sub, _ = sampler.sample(embeddings, ds)
            logger.info(f"Sampled to {emb_sub.shape[0]} samples using {type(sampler).__name__}")

    module = kwargs.get("module", None)

    # Log dataset capabilities
    from manylatents.data.capabilities import log_capabilities
    log_capabilities(ds_sub)

    metric_cfgs = flatten_and_unroll_metrics(cfg.metrics) if cfg.metrics is not None else {}

    # Pre-warm cache with optimal k values
    knn_cache_dir = getattr(cfg, "cache_dir", None)
    cache = prewarm_cache(metric_cfgs, emb_sub, ds_sub, module, knn_cache_dir=knn_cache_dir)

    results: dict[str, Any] = {}
    for metric_name, metric_cfg in metric_cfgs.items():
        metric_fn = hydra.utils.instantiate(metric_cfg)

        raw_result = metric_fn(
            embeddings=emb_sub,
            dataset=ds_sub,
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
            results[metric_name] = raw_result

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

            # Attach trajectories if the algorithm produced them
            traj = getattr(algorithm, "trajectories", None)
            if traj is not None:
                if isinstance(traj, torch.Tensor):
                    traj = traj.detach().cpu().numpy()
                embeddings["trajectories"] = traj
                logger.info(f"Trajectories attached: shape={traj.shape}")

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


def run_pipeline(cfg: DictConfig, input_data_holder: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Execute a sequential multi-step pipeline workflow.

    This orchestrator enables depth workflows (sequential chaining) where
    the output embeddings of step N become the input data for step N+1.
    Example: PCA (1000→50) → PHATE (50→2)

    Args:
        cfg: Hydra configuration containing a 'pipeline' key with list of steps

    Returns:
        Dictionary with keys: embeddings, label, metadata, scores (from final step)
    """
    setup_logging(debug=cfg.debug, log_level=getattr(cfg, "log_level", "warning"))
    logger.info("Pipeline Config:\n" + OmegaConf.to_yaml(cfg))

    if not hasattr(cfg, 'pipeline') or cfg.pipeline is None or len(cfg.pipeline) == 0:
        raise ValueError("No pipeline configuration found. Use run_algorithm() for single runs.")

    # Determine if WandB should be disabled using centralized check
    wandb_disabled = should_disable_wandb(cfg) or wandb is None

    # Initialize WandB based on configuration (respects logger=None, debug=True, and WANDB_MODE)
    if wandb is not None:
        if wandb_disabled:
            logger.info("Initializing WandB in disabled mode")
            wandb_context = wandb.init(
                project=cfg.project,
                name=cfg.name,
                config=OmegaConf.to_container(cfg, resolve=True),
                mode="disabled",
                dir=os.environ.get("WANDB_DIR", "logs"),
            )
        else:
            logger.info("Initializing WandB in online mode")
            wandb_context = wandb.init(
                project=cfg.project,
                name=cfg.name,
                config=OmegaConf.to_container(cfg, resolve=True),
                mode="online",
                dir=os.environ.get("WANDB_DIR", "logs"),
            )
    else:
        from contextlib import nullcontext
        wandb_context = nullcontext()

    with wandb_context as run:
        lightning.seed_everything(cfg.seed, workers=True)

        # --- One-time setup: Create initial datamodule, trainer, callbacks ---
        logger.info("Setting up pipeline infrastructure...")

        # Initial datamodule (loads original data)
        initial_datamodule = instantiate_datamodule(cfg, input_data_holder)
        initial_datamodule.setup()

        # Setup callbacks (shared across all steps)
        trainer_cb_cfg = cfg.trainer.get("callbacks", {})
        embedding_cb_cfg = cfg.get("callbacks", {}).get("embedding", {})
        lightning_cbs, embedding_cbs = instantiate_callbacks(trainer_cb_cfg, embedding_cb_cfg)

        if not embedding_cbs:
            logger.info("No embedding callbacks configured.")

        # Setup loggers (use same logic as run_algorithm for consistency)
        loggers = []
        if not wandb_disabled and cfg.logger is not None:
            for lg_conf in cfg.trainer.get("logger", {}).values():
                loggers.append(hydra.utils.instantiate(lg_conf))
            logger.info(f"Trainer loggers enabled: {len(loggers)} logger(s)")
        else:
            logger.info("Trainer loggers disabled (WandB disabled)")

        # Setup trainer (shared across all steps)
        trainer = instantiate_trainer(
            cfg,
            lightning_callbacks=lightning_cbs,
            loggers=loggers,
        )

        # --- Load initial data ---
        logger.info("Loading initial data from datamodule...")
        train_loader = initial_datamodule.train_dataloader()
        test_loader = initial_datamodule.test_dataloader()
        field_index, data_source = determine_data_source(train_loader)

        initial_train_tensor = torch.cat([b[field_index].cpu() for b in train_loader], dim=0)
        initial_test_tensor = torch.cat([b[field_index].cpu() for b in test_loader], dim=0)

        # Extract labels for supervised LatentModules (if available)
        initial_train_labels = None
        train_dataset = getattr(initial_datamodule, "train_dataset", None)
        if train_dataset is not None and hasattr(train_dataset, "get_labels"):
            labels = train_dataset.get_labels()
            if labels is not None:
                initial_train_labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
                logger.info(f"Extracted {len(initial_train_labels)} training labels for supervised learning")

        logger.info(
            f"Initial data from {data_source}:\n"
            f"  Train: {initial_train_tensor.shape}\n"
            f"  Test: {initial_test_tensor.shape}"
        )

        # Track current embeddings and datamodule
        current_train_tensor = initial_train_tensor
        current_test_tensor = initial_test_tensor
        current_train_labels = initial_train_labels
        current_datamodule = initial_datamodule
        final_algorithm = None

        # --- Loop through pipeline steps ---
        step_snapshots = []
        step_times = []
        t_total_start = time.perf_counter()
        logger.info(f"Starting pipeline with {len(cfg.pipeline)} steps...\n")

        for step_idx, step in enumerate(cfg.pipeline):
            logger.info(f"{'='*60}")
            logger.info(f"PIPELINE STEP {step_idx + 1}/{len(cfg.pipeline)}")
            logger.info(f"{'='*60}")

            # Merge step overrides with global config
            # Need to disable struct mode to allow new keys
            step_overrides = step.get('overrides', {})
            OmegaConf.set_struct(cfg, False)
            step_cfg = OmegaConf.merge(cfg, step_overrides)
            OmegaConf.set_struct(step_cfg, True)

            step_name = step.get('name', f'step_{step_idx}')
            logger.info(f"Step name: {step_name}")
            logger.info(f"Input shape: {current_test_tensor.shape}")

            # Instantiate step-specific algorithm
            if hasattr(step_cfg.algorithms, 'latent') and step_cfg.algorithms.latent is not None:
                algorithm = instantiate_algorithm(step_cfg.algorithms.latent, current_datamodule)
            elif hasattr(step_cfg.algorithms, 'lightning') and step_cfg.algorithms.lightning is not None:
                algorithm = instantiate_algorithm(step_cfg.algorithms.lightning, current_datamodule)
            else:
                raise ValueError(f"No algorithm specified in pipeline step {step_idx + 1}")

            logger.info(f"Algorithm: {type(algorithm).__name__}")

            # Execute the step with current tensors
            t_step_start = time.perf_counter()
            latents = execute_step(
                algorithm=algorithm,
                train_tensor=current_train_tensor,
                test_tensor=current_test_tensor,
                trainer=trainer,
                cfg=step_cfg,
                datamodule=current_datamodule,
                train_labels=current_train_labels,
            )
            step_time = time.perf_counter() - t_step_start
            step_times.append(step_time)

            if latents is None:
                raise RuntimeError(
                    f"Pipeline step {step_idx + 1} ({step_name}) failed to produce output embeddings"
                )

            # Convert to tensor for next step
            if isinstance(latents, np.ndarray):
                current_test_tensor = torch.from_numpy(latents).float()
            else:
                current_test_tensor = latents

            # For sequential pipelines, use same data for train and test in subsequent steps
            current_train_tensor = current_test_tensor
            # Labels only apply to original data; clear for subsequent steps
            current_train_labels = None

            # Note: For next steps, we'd ideally create a PrecomputedDataModule from latents
            # For now, we keep using initial_datamodule (algorithm only needs it for metadata)
            # Future enhancement: Create in-memory PrecomputedDataModule

            logger.info(f"Step {step_idx + 1} complete. Output shape: {current_test_tensor.shape}\n")

            snapshot_np = (current_test_tensor.detach().cpu().numpy()
                           if isinstance(current_test_tensor, torch.Tensor)
                           else np.asarray(current_test_tensor))
            step_snapshots.append({
                "step_index": step_idx,
                "step_name": step_name,
                "algorithm": type(algorithm).__name__,
                "output_shape": list(snapshot_np.shape),
                "embedding": snapshot_np,
                "step_time": step_time,
            })

            # Track final algorithm for metadata
            final_algorithm = algorithm

        # --- Final wrapping: Evaluate and callback on final embeddings ---
        logger.info("Pipeline execution complete. Preparing final outputs...")

        final_latents = current_test_tensor
        if isinstance(final_latents, torch.Tensor):
            final_latents = final_latents.detach().cpu().numpy()

        embeddings = {
            "embeddings": final_latents,
            "label": getattr(getattr(initial_datamodule, "test_dataset", None), "get_labels", lambda: None)(),
            "metadata": {
                "source": "pipeline",
                "num_steps": len(cfg.pipeline),
                "final_algorithm_type": type(final_algorithm).__name__ if final_algorithm else "unknown",
                "input_shape": initial_test_tensor.shape,
                "output_shape": final_latents.shape,
            },
            "step_snapshots": step_snapshots,
        }

        # Evaluate final embeddings
        logger.info("Evaluating final embeddings from pipeline...")
        t_eval_start = time.perf_counter()
        embeddings["scores"] = evaluate(
            embeddings,
            cfg=cfg,
            datamodule=initial_datamodule,
            module=final_algorithm if isinstance(final_algorithm, LatentModule) else None
        )
        eval_time = time.perf_counter() - t_eval_start
        total_time = time.perf_counter() - t_total_start

        embeddings["metadata"]["step_times"] = step_times
        embeddings["metadata"]["eval_time"] = eval_time
        embeddings["metadata"]["total_time"] = total_time

        # --- Callback processing ---
        callback_outputs = {}
        if embeddings and embedding_cbs:
            logger.info("Running embedding callbacks...")
            for cb in embedding_cbs:
                cb_result = cb.on_latent_end(dataset=initial_datamodule.test_dataset, embeddings=embeddings)
                # Merge callback outputs into the main callback_outputs dict
                if isinstance(cb_result, dict):
                    callback_outputs.update(cb_result)
                    logger.info(f"Callback {cb.__class__.__name__} returned: {list(cb_result.keys())}")

        # Add callback outputs to embeddings dict if any were generated
        if callback_outputs:
            embeddings['callback_outputs'] = callback_outputs
            logger.info(f"Added callback outputs to embeddings: {list(callback_outputs.keys())}")

        logger.info("Pipeline workflow complete.")

        # Auto-log scores to wandb if active (works online and offline)
        if wandb is not None and wandb.run and embeddings.get("scores"):
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

        if wandb is not None and wandb.run:
            wandb.finish()

        return embeddings