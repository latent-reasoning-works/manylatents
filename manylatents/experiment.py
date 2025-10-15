import functools
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import lightning
import numpy as np
import torch
import wandb
from lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
)
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from manylatents.algorithms.latent_module_base import LatentModule
from manylatents.callbacks.embedding.base import EmbeddingCallback
from manylatents.utils.data import subsample_data_and_dataset, determine_data_source
from manylatents.utils.metrics import flatten_and_unroll_metrics
from manylatents.utils.utils import check_or_make_dirs, load_precomputed_embeddings, setup_logging

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

@evaluate.register(dict)
def evaluate_embeddings(
    EmbeddingOutputs: dict,
    *,
    cfg: DictConfig,
    datamodule,
    **kwargs,
) -> dict:
    if EmbeddingOutputs is None or EmbeddingOutputs.get("embeddings") is None:
        logger.warning("No embeddings available for evaluation.")
        return {}

    embeddings = EmbeddingOutputs.get("embeddings")
 
    # Handle different datamodule types - some store mode directly, others in hparams
    mode = getattr(datamodule, 'mode', None) or getattr(datamodule.hparams, 'mode', 'full')
    
    if mode == "split":
        ds = datamodule.test_dataset
    else:
        ds = datamodule.train_dataset ## defaults to full dataset on full runs

    logger.info(f"Reference data shape: {ds.data.shape}")
    logger.info(f"Computing embedding metrics for {ds.data.shape[0]} samples.")

    #subsample in case dataset is too large
    subsample_fraction = cfg.metrics.get("subsample_fraction", None) if cfg.metrics is not None else None
    if subsample_fraction is not None:
        ds_sub, emb_sub = subsample_data_and_dataset(ds, embeddings, subsample_fraction)
        logger.info(f"Subsampled dataset to {emb_sub.shape[0]} samples.")
    else:
        ds_sub, emb_sub = ds, embeddings

    module = kwargs.get("module", None)

    metric_cfgs = flatten_and_unroll_metrics(cfg.metrics) if cfg.metrics is not None else {}

    results: dict[str, float] = {}
    for metric_name, metric_cfg in metric_cfgs.items():
        metric_fn = hydra.utils.instantiate(metric_cfg)
        results[metric_name] = metric_fn(
            embeddings=emb_sub,
            dataset=ds_sub,
            module=module,
        )
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
    datamodule: Optional[LightningDataModule] = None
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

    Returns:
        The computed latent embeddings as a tensor
    """
    latents = None

    # --- Compute latents based on algorithm type ---
    if isinstance(algorithm, LatentModule):
        # LatentModule: fit/transform pattern
        algorithm.fit(train_tensor)
        latents = algorithm.transform(test_tensor)
        logger.info(f"LatentModule embedding shape: {latents.shape}")

    elif isinstance(algorithm, LightningModule):
        # LightningModule: training or eval-only

        # Handle eval-only mode with pretrained checkpoint
        if cfg.eval_only and hasattr(cfg, 'pretrained_ckpt') and cfg.pretrained_ckpt:
            logger.info(f"Loading pretrained model from {cfg.pretrained_ckpt}")
            algorithm = LightningModule.load_from_checkpoint(cfg.pretrained_ckpt)

        # Training phase (if not eval_only)
        if not cfg.eval_only:
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
    setup_logging(debug=cfg.debug)
    logger.info("Final Config:\n" + OmegaConf.to_yaml(cfg))

    with wandb.init(
        project=cfg.project,
        name=cfg.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="disabled" if cfg.debug else "online",
    ) as run:
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
        trainer_cb_cfg   = cfg.trainer.get("callbacks")
        embedding_cb_cfg = cfg.get("callbacks.embedding")

        lightning_cbs, embedding_cbs = instantiate_callbacks(
            trainer_cb_cfg,
            embedding_cb_cfg
        )

        if not embedding_cbs:
            logger.info("No embedding callbacks configured; skip embedding‐level hooks.")

        # --- Loggers ---
        loggers = []
        if not cfg.debug:
            for lg_conf in cfg.trainer.get("logger", {}).values():
                loggers.append(hydra.utils.instantiate(lg_conf))

        # --- Trainer ---
        trainer = instantiate_trainer(
            cfg,
            lightning_callbacks=lightning_cbs,
            loggers=loggers,
        )

        logger.info("Starting algorithm execution...")

        # --- Algorithm Execution ---
        embeddings: Dict[str, Any] = {}

        if cfg.eval_only:
            logger.info("Evaluation-only mode: Loading precomputed latent outputs.")
            embeddings = load_precomputed_embeddings(cfg)
        else:
            # Unroll train and test dataloaders to obtain tensors
            train_tensor = torch.cat([b[field_index].cpu() for b in train_loader], dim=0)
            test_tensor  = torch.cat([b[field_index].cpu() for b in test_loader],  dim=0)

            logger.info(
                f"Running algorithm on {data_source}:\n"
                f"Train tensor shape: {train_tensor.shape}\n"
                f"Test tensor shape: {test_tensor.shape}\n"
                f"Algorithm: {type(algorithm).__name__}"
            )

            # Execute the algorithm step
            latents = execute_step(
                algorithm=algorithm,
                train_tensor=train_tensor,
                test_tensor=test_tensor,
                trainer=trainer,
                cfg=cfg,
                datamodule=datamodule
            )

            # --- Unified embedding wrapping ---
            if latents is not None:
                embeddings = {
                    "embeddings": latents,
                    "label": getattr(getattr(datamodule, "test_dataset", None), "get_labels", lambda: None)(),
                    "metadata": {
                        "source": "single_algorithm",
                        "algorithm_type": type(algorithm).__name__,
                        "data_shape": test_tensor.shape,
                    },
                }

                # Evaluate embeddings
                logger.info(f"Evaluating embeddings from {type(algorithm).__name__}...")
                embeddings["scores"] = evaluate(
                    embeddings,
                    cfg=cfg,
                    datamodule=datamodule,
                    module=algorithm if isinstance(algorithm, LatentModule) else None
                )

        # --- Callback processing ---
        if embeddings and embedding_cbs:
            for cb in embedding_cbs:
                cb.on_latent_end(dataset=datamodule.test_dataset, embeddings=embeddings)

        logger.info("Experiment complete.")

        if wandb.run:
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
    setup_logging(debug=cfg.debug)
    logger.info("Pipeline Config:\n" + OmegaConf.to_yaml(cfg))

    if not hasattr(cfg, 'pipeline') or cfg.pipeline is None or len(cfg.pipeline) == 0:
        raise ValueError("No pipeline configuration found. Use run_algorithm() for single runs.")

    with wandb.init(
        project=cfg.project,
        name=cfg.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="disabled" if cfg.debug else "online",
    ) as run:
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

        # Setup loggers
        loggers = []
        if not cfg.debug:
            for lg_conf in cfg.trainer.get("logger", {}).values():
                loggers.append(hydra.utils.instantiate(lg_conf))

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

        logger.info(
            f"Initial data from {data_source}:\n"
            f"  Train: {initial_train_tensor.shape}\n"
            f"  Test: {initial_test_tensor.shape}"
        )

        # Track current embeddings and datamodule
        current_train_tensor = initial_train_tensor
        current_test_tensor = initial_test_tensor
        current_datamodule = initial_datamodule
        final_algorithm = None

        # --- Loop through pipeline steps ---
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
            latents = execute_step(
                algorithm=algorithm,
                train_tensor=current_train_tensor,
                test_tensor=current_test_tensor,
                trainer=trainer,
                cfg=step_cfg,
                datamodule=current_datamodule
            )

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

            # Note: For next steps, we'd ideally create a PrecomputedDataModule from latents
            # For now, we keep using initial_datamodule (algorithm only needs it for metadata)
            # Future enhancement: Create in-memory PrecomputedDataModule

            logger.info(f"Step {step_idx + 1} complete. Output shape: {current_test_tensor.shape}\n")

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
        }

        # Evaluate final embeddings
        logger.info("Evaluating final embeddings from pipeline...")
        embeddings["scores"] = evaluate(
            embeddings,
            cfg=cfg,
            datamodule=initial_datamodule,
            module=final_algorithm if isinstance(final_algorithm, LatentModule) else None
        )

        # --- Callback processing ---
        if embeddings and embedding_cbs:
            logger.info("Running embedding callbacks...")
            for cb in embedding_cbs:
                cb.on_latent_end(dataset=initial_datamodule.test_dataset, embeddings=embeddings)

        logger.info("Pipeline workflow complete.")

        if wandb.run:
            wandb.finish()

        return embeddings