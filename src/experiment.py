import functools
import logging
from typing import Any, Optional, Tuple, Union

import hydra
import numpy as np
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.algorithms.dimensionality_reduction import DimensionalityReductionModule
from src.utils.data import DummyDataModule
from src.utils.utils import check_or_make_dirs
from src.metrics.handler import MetricsHandler


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
    embeddings: Optional[np.ndarray] = None,
):
    """
    Train the model using PyTorch Lightning Trainer.
    """
    if model is None:
        raise ValueError("No model was instantiated. Check your config under 'algorithm.network'.")
    
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
    datamodule: Union[LightningDataModule, DataLoader],
    embeddings: Optional[np.ndarray] = None,
    **kwargs,
) -> Tuple[str, Optional[float], dict]:
    """
    Evaluate the DR algorithm and, if configured, compute additional metrics using a metrics handler.
    """
    if embeddings is None:
        raise ValueError("No embeddings were computed. Check your DR algorithm configuration.")
    
    # Compute the base error metric from the DR module if available
    error_metric = algorithm.evaluate(embeddings) 
    
    # Optionally compute additional metrics via the metrics handler if "metrics" is in the config.
    metrics_results = {}
    if "metrics" in cfg:
        try:
            metrics_handler = MetricsHandler(cfg.metrics)
            # Expect the metadata to be provided via kwargs.
            metadata = kwargs.get("original_data", None)
            if metadata is None:
                logger.warning("No metadata provided for metrics computation. Skipping additional metrics.")
            else:
                metrics_results = metrics_handler.compute_all(original=original_data, embedded=embeddings)
        except Exception as e:
            logger.error(f"Failed to compute additional metrics: {e}")
    
    return "error", error_metric, metrics_results


@evaluate.register(LightningModule)
def evaluate_lightningmodule(
    algorithm: LightningModule,
    *,
    cfg: DictConfig,
    trainer: Trainer,
    datamodule: Union[LightningDataModule, DataLoader],
    embeddings: Optional[np.ndarray] = None,
) -> Tuple[str, Optional[float], dict]:
    """
    Evaluate the model on the test set.
    Sets the model to evaluation mode before testing.
    """
    if algorithm is None:
        raise ValueError("No model was instantiated. Check your config under 'algorithm.network'.")

    results = trainer.test(model=algorithm, datamodule=datamodule)
    if not results:
        return "fail", 9999.0, {}

    # Suppose we want 'loss' or 'accuracy' from results
    metrics = results[0]
    if "test/accuracy" in metrics:
        acc = metrics["test/accuracy"]
        return "accuracy", (1.0 - acc), metrics
    elif "test/loss" in metrics:
        return "loss", metrics["test/loss"], metrics
    else:
        return "unknown", None, metrics
