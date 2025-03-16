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
    ## CURRENTLY DOESN'T USE EMBEDDINGS, REVISE
    ## BEFORE ADDING TRAINING SUPPORT
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
) -> dict:
    """
    Evaluate the DR algorithm and return a dictionary of metrics.
    This function ensures that the original high-dimensional data is passed
    to the evaluate method.
    """
    # Try to get original data from datamodule
    original_data = getattr(datamodule.train_dataset, "original_data", None)
    logger.info(f"Original data shape: {original_data.shape if original_data is not None else None}")
    # If datamodule does not provide it, attempt to use one passed via kwargs.
    if original_data is None and "original_data" in kwargs:
        original_data = kwargs["original_data"]
    
    # If still not available, raise an error.
    if original_data is None:
        raise ValueError("No original data available for evaluation.")
    
    # Call the module's evaluate method with original high-dimensional data.
    metrics = algorithm.evaluate(original_data, embeddings)
    return metrics

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
