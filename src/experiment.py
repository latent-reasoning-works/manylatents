import functools
import logging
from typing import Any, Optional, Tuple, Union

import hydra
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.utils.data import DummyDataModule, subsample_data_and_dataset
from src.utils.utils import check_or_make_dirs

logger = logging.getLogger(__name__)

def instantiate_datamodule(cfg: DictConfig) -> LightningDataModule:
    check_or_make_dirs(cfg.cache_dir)
    logger.info(f"Cache directory ensured at: {cfg.cache_dir}")

    if cfg.data.get("debug", False):
        logger.info("DEBUG MODE: Using a dummy datamodule with limited data.")
        dummy_data = torch.randn(100, 10)
        dummy_labels = torch.zeros(100)
        dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
        return DummyDataModule(
            dataset=dataset, 
            batch_size=cfg.data.batch_size, 
            num_workers=cfg.data.num_workers
        )

    datamodule_cfg = {k: v for k, v in cfg.data.items() if k != "debug"}
    dm = hydra.utils.instantiate(datamodule_cfg)
    dm.setup()
    return dm

def instantiate_trainer(cfg: DictConfig) -> Trainer:
    """
    Dynamically instantiate the PyTorch Lightning Trainer from the config.
    Handles callbacks and loggers if specified.
    """
    trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)

    # Load callbacks & logger from separate configs
    callbacks = hydra.utils.instantiate(cfg.callbacks) if "callbacks" in cfg else []
    loggers = hydra.utils.instantiate(cfg.logger) if "logger" in cfg and cfg.logger is not None else None

    # Remove from trainer config to avoid duplicate passing
    trainer_config.pop("callbacks", None)
    trainer_config.pop("logger", None)

    return hydra.utils.instantiate(trainer_config, callbacks=callbacks, logger=loggers)
    
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
    
    if datamodule.mode == "split":
        ds = datamodule.test_dataset
    else:
        ds = datamodule.train_dataset ## defaults to full datset on full runs

    # Use original_data as the reference if available;
    # if not, fallback to precomputed_embeddings.
    if ds.original_data is not None:
        reference = ds.original_data #[ds.split_indices[ds.data_split]]
    elif ds.precomputed_embeddings is not None:
        logger.warning("No raw data provided; using precomputed embeddings as reference for evaluation.")
        reference = ds.precomputed_embeddings #[ds.split_indices[ds.data_split]]
    else:
        raise ValueError("No valid reference data available for evaluation.")

    logger.info(f"Reference data shape: {reference.shape}")
    logger.info(f"Computing embedding metrics for {reference.shape[0]} samples.")

    metrics = {}
    all_metrics_cfg = cfg.metrics
    ds_metrics_cfg = all_metrics_cfg.get("dataset", {})
    ds_subsample_fraction = all_metrics_cfg.get("subsample_fraction", None)

    if ds_subsample_fraction is not None:
        ds_subsampled, embeddings_subsampled = subsample_data_and_dataset(ds, EmbeddingOutputs.get("embeddings"), ds_subsample_fraction)
        logger.info(f"Subsampled dataset to {embeddings_subsampled.shape[0]} samples for dataset-level metrics.")
    else:
        ds_subsampled, embeddings_subsampled = ds, EmbeddingOutputs.get("embeddings")

    for metric_name, metric_config in ds_metrics_cfg.items():
        metric_fn = hydra.utils.instantiate(metric_config)
        result = metric_fn(ds_subsampled, embeddings_subsampled)
        metrics[metric_name] = result

    module_metrics_cfg = all_metrics_cfg.get("embedding", {})
    for metric_name, metric_config in module_metrics_cfg.items():
        metric_fn = hydra.utils.instantiate(metric_config)
        result = metric_fn(ds, EmbeddingOutputs.get("embeddings"))
        metrics[metric_name] = result

    return metrics

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
    model_metrics_cfg = cfg.metrics.get("model", {})

    for metric_key, metric_params in model_metrics_cfg.items():
        if metric_params.get("enabled", True):
            metric_fn = hydra.utils.instantiate(metric_params)
            # Let any errors during metric computation propagate.
            name, value = metric_fn(algorithm, test_results=base_metrics)
            custom_metrics[name] = value

    combined_metrics = {**base_metrics, **custom_metrics}
    error_value = next(iter(combined_metrics.values())) if combined_metrics else None
    return combined_metrics, error_value
