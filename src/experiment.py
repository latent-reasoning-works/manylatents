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
        return DummyDataModule(
            dataset=dataset, 
            batch_size=cfg.datamodule.batch_size, 
            num_workers=cfg.datamodule.num_workers
        )

    datamodule_cfg = {k: v for k, v in cfg.datamodule.items() if k != "debug"}
    dm = hydra.utils.instantiate(datamodule_cfg)
    dm.setup()
    return dm

def instantiate_algorithm(
    cfg: DictConfig,
    datamodule: Optional[LightningDataModule] = None,
) -> Tuple[Optional[np.ndarray], Optional[torch.nn.Module]]:
    """
    Instantiates the algorithms specified in the configuration.
    
    The function returns a tuple: (embeddings, model).
    
    - If a Dimensionality Reduction (DR) algorithm is configured, 
      the DR method is run immediately on the training data to compute embeddings.
    - If only a Neural Network (NN) is configured, the function returns the NN model
      and leaves the embeddings as None (to be extracted later via the model's forward pass).
    - If both DR and NN are specified, the DR embeddings are computed first and can be used
      as inputs for the NN training process.
    
    Args:
        cfg (DictConfig): Hydra configuration.
        datamodule (Optional[LightningDataModule]): Data module for data loading.
    
    Returns:
        Tuple[Optional[np.ndarray], Optional[torch.nn.Module]]:
            - embeddings: Computed embeddings from the DR algorithm, or None if using NN only.
            - model: Instantiated neural network model, or None if only DR is used.
    """
    embeddings_result: Optional[np.ndarray] = None
    model: Optional[torch.nn.Module] = None

    if "dimensionality_reduction" in cfg.algorithm and cfg.algorithm.dimensionality_reduction is not None:
        dr_cfg = cfg.algorithm.dimensionality_reduction
        if "_target_" not in dr_cfg:
            raise ValueError("Missing _target_ in dimensionality_reduction config")

        logger.info(f"Instantiating Dimensionality Reduction: {dr_cfg._target_.split('.')[-1]}")
        dr_algorithm = hydra.utils.instantiate(dr_cfg)

        if datamodule is None:
            raise ValueError("DataModule must be provided for Dimensionality Reduction.")

        data_loader = datamodule.train_dataloader()
        data_list = [inputs.cpu().numpy() for inputs, _ in data_loader]
        data_np = np.concatenate(data_list, axis=0)
        logger.debug(f"Data shape for embedding: {data_np.shape}")

        dr_algorithm.fit(torch.tensor(data_np))
        embeddings_result = dr_algorithm.transform(torch.tensor(data_np))
        logger.info(f"Embedding completed with shape: {embeddings_result.shape}")

        if "dimensionality_reduction" in cfg.callbacks:
            callback_cfg = cfg.callbacks.dimensionality_reduction
            dr_callback = hydra.utils.instantiate(callback_cfg)  # Hydra resolves save_dir
            logger.info(f"Instantiated callback: {dr_callback.__class__.__name__}")
            
            if hasattr(dr_callback, "on_dr_end") and callable(dr_callback.on_dr_end):
                logger.info("Calling on_dr_end() to save embeddings...")
                dr_callback.on_dr_end(embeddings_result)
            else:
                logger.warning("Callback has no on_dr_end(). Skipping.")

    if "network" in cfg.algorithm and cfg.algorithm.network is not None:
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
    Sets the model to evaluation mode before testing.
    """
    if model is None:
        raise ValueError("No model was instantiated. Check your config under 'algorithm.network'.")

    model.eval()  # Set the model to evaluation mode for inference
    trainer.test(model, datamodule=datamodule)