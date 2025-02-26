import logging
from typing import Optional, Tuple, Union

import hydra
import numpy as np
import torch
from lightning import LightningDataModule, Trainer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

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
        return DummyDataModule(dataset=dataset, 
                               batch_size=cfg.datamodule.batch_size, 
                               num_workers=cfg.datamodule.num_workers
                               )

    datamodule_cfg = {k: v for k, v in cfg.datamodule.items() if k != "debug"}
    dm = hydra.utils.instantiate(datamodule_cfg)
    dm.setup()
    return dm

    if "dataloader" in cfg and cfg.dataloader is not None:
        raise ValueError("Use a LightningDataModule instead of a raw DataLoader.")

    raise ValueError("No valid 'datamodule' found in the config.")

def instantiate_algorithm(
    cfg: DictConfig,
    datamodule: Optional[LightningDataModule] = None,
) -> Tuple[Optional[np.ndarray], Optional[torch.nn.Module]]:
    """
    Instantiates the necessary algorithms based on the configuration.

    Scenarios:
    - Only Embedding
    - Only Network
    - Both Embedding and Network

    Args:
        cfg (DictConfig): Hydra configuration.
        datamodule (Optional[LightningDataModule]): Data module for data loading.

    Returns:
        Tuple[Optional[np.ndarray], Optional[torch.nn.Module]]: 
            - embeddings_result: Embeddings if DR is performed.
            - model: Instantiated network model.
    """
    embeddings_result = None
    model = None

    if "dimensionality_reduction" in cfg.algorithm:
        dr_cfg = cfg.algorithm.dimensionality_reduction
        if "_target_" not in dr_cfg:
            raise ValueError("Missing _target_ in dimensionality_reduction config")

        logger.info(f"Instantiating dimensionality reduction: {dr_cfg._target_.split('.')[-1]}")
        dr_algorithm = hydra.utils.instantiate(dr_cfg)

        if datamodule is None:
            raise ValueError("DataModule must be provided for dimensionality reduction.")

        # Load data from datamodule
        data_loader = datamodule.train_dataloader()
        data = [inputs.cpu().numpy() for inputs, _ in data_loader]
        data_np = np.concatenate(data, axis=0)
        logger.debug(f"Data shape for embedding: {data_np.shape}")

        dr_algorithm.fit(torch.tensor(data_np))
        embeddings_result = dr_algorithm.transform(torch.tensor(data_np))
        logger.info(f"Embedding completed with shape: {embeddings_result.shape}")

    if "network" in cfg.algorithm:
        model_cfg = cfg.algorithm.network
        if "_target_" not in model_cfg:
            raise ValueError("Missing _target_ in network config")

        logger.info(f"Instantiating Neural Network: {model_cfg._target_}")
        model = hydra.utils.instantiate(model_cfg)

    return embeddings_result, model

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
    
def evaluate_model(
    cfg: DictConfig,
    trainer: Trainer,
    datamodule: Union[LightningDataModule, DataLoader],
    model: torch.nn.Module,
    embeddings: Optional[np.ndarray] = None,
):
    """
    Evaluate the model on the test set.
    """
    if model is None:
        raise ValueError("No model was instantiated. Check your config under 'algorithm.network'.")

    trainer.test(model, datamodule=datamodule)  
