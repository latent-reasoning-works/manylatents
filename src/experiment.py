import functools
import logging
from typing import Any, Optional, Tuple, Union

import hydra
import numpy as np
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.fabric.utilities.exceptions import MisconfigurationException
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.algorithms.dimensionality_reduction import DimensionalityReductionModule
from src.utils.data import DummyDataModule, subsample_data_and_dataset
from src.utils.utils import check_or_make_dirs

logger = logging.getLogger(__name__)

def instantiate_datamodule(cfg: DictConfig) -> LightningDataModule:
    check_or_make_dirs(cfg.cache_dir)
    logger.info(f"Cache directory ensured at: {cfg.cache_dir}")

    if cfg.datamodule.get("debug", False):
        logger.info("DEBUG MODE: Using a dummy datamodule with limited data.")
        dummy_data = torch.randn(100, 10)
        dummy_labels = torch.zeros(100)
        dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
        return DummyDataModule(
            dataset=dataset, 
            batch_size=cfg.datamodule.batch_size, 
            num_workers=cfg.datamodule.num_workers
        )

    datamodule_cfg = {k: v for k, v in cfg.datamodule.items() if k != "debug"}
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

def train_model(
    cfg: DictConfig,
    trainer: Trainer,
    datamodule: Union[LightningDataModule, DataLoader],
    model: torch.nn.Module,
    embeddings: Optional[np.ndarray] = None,  # Allow None
):
    """
    Train the model using PyTorch Lightning Trainer.
    """
    if model is None:
        raise ValueError("No model was instantiated. Check your config under 'algorithm.network'.")

    logger.info(f"Training model with embeddings: {embeddings is not None}") 
    trainer.fit(model, datamodule=datamodule)

    
@functools.singledispatch
def evaluate(algorithm: Any, /, **kwargs) -> Tuple[str, Optional[float], dict]:
    """Evaluates the algorithm.

    Returns the name of the 'error' metric for this run, its value, and a dict of metrics.
    """
    raise NotImplementedError(
        f"There is no registered handler for evaluating algorithm {algorithm} of type "
        f"{type(algorithm)}! (kwargs: {kwargs})"
    )
    
@evaluate.register(DimensionalityReductionModule)
def evaluate_dr(
    algorithm: DimensionalityReductionModule,
    *,
    cfg: DictConfig,
    datamodule,
    embeddings: Optional[np.ndarray] = None,
    **kwargs,
) -> dict:
    
    if datamodule.mode == "split":
        ds = datamodule.test_dataset
    else:
        ds = datamodule.train_dataset

    # Subset the original data using the split indices for consistency.
    original_data = ds.original_data[ds.split_indices[ds.data_split]]
        
    if original_data is None:
        raise ValueError("No original data available for evaluation.")
    
    logger.info(f"Original data shape: {original_data.shape}")
    logger.info(f"Computing DR metrics for {original_data.shape[0]} samples.")
    
    metrics = {}
    
    ds_metrics_cfg = cfg.metrics.get("dataset", {})
    ds_subsample_fraction = ds_metrics_cfg.get("subsample_fraction", None)
    if ds_subsample_fraction is not None:
        ds_subsampled, embeddings_subsampled = subsample_data_and_dataset(ds, embeddings, ds_subsample_fraction)
        logger.info(f"Subsampled dataset to {embeddings_subsampled.shape[0]} samples for dataset-level metrics.")
    else:
        ds_subsampled, embeddings_subsampled = ds, embeddings
    
    for metric_name, metric_params in ds_metrics_cfg.items():
        if metric_name == "subsample_fraction":
            continue
        if metric_params.get("enabled", True):
            metric_fn = hydra.utils.instantiate(metric_params)
            metrics[metric_name] = metric_fn(ds_subsampled, embeddings_subsampled)
    
    module_metrics_cfg = cfg.metrics.get("module", {})
    for metric_name, metric_params in module_metrics_cfg.items():
        if metric_params.get("enabled", True):
            metric_fn = hydra.utils.instantiate(metric_params)
            metrics[metric_name] = metric_fn(ds, embeddings)
    
    return metrics

@evaluate.register(LightningModule)
def evaluate_lightningmodule(
    algorithm: LightningModule,
    *,
    cfg: DictConfig,
    trainer: Trainer,
    datamodule: Union[LightningDataModule, DataLoader],
    embeddings: Optional[any] = None,
    **kwargs,
) -> dict:
    """
    Evaluate the LightningModule on the test set and compute additional custom metrics.
    
    If the model does not implement a `test_step()`, this function will skip evaluation and return an empty dictionary.
    
    Returns:
        A dictionary mapping each metric name to its computed value.
    """
    # Optionally, you can check before calling trainer.test
    if not hasattr(algorithm, "test_step"):
        logger.info("Model does not define a test_step() method; skipping evaluation.")
        return {}

    try:
        results = trainer.test(model=algorithm, datamodule=datamodule)
    except MisconfigurationException as e:
        logger.info(f"Skipping evaluation due to misconfiguration: {e}")
        return {}
    
    if not results:
        return {}
    base_metrics = results[0]

    custom_metrics = {}
    model_metrics_cfg = cfg.metrics.get("model", {})
    for metric_key, metric_params in model_metrics_cfg.items():
        if metric_params.get("enabled", True):
            try:
                metric_fn = hydra.utils.instantiate(metric_params)
                # Each metric_fn should return a tuple (name, value)
                name, value = metric_fn(algorithm, test_results=base_metrics)
                custom_metrics[name] = value
            except Exception as e:
                logger.error(f"Error computing metric '{metric_key}': {e}")
    
    combined_metrics = {**base_metrics, **custom_metrics}
    return combined_metrics