import logging
from typing import Optional, Tuple, Union

import hydra
import numpy as np
import torch
from lightning import Trainer
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.utils.utils import check_or_make_dirs

logger = logging.getLogger(__name__)

def instantiate_datamodule(cfg: DictConfig) -> Union[LightningDataModule, DataLoader]:
    """
    Dynamically instantiate the data module (or dataloader) from the config.
    """
    check_or_make_dirs(cfg.paths.cache_dir)
    logger.info(f"Cache directory ensured at: {cfg.paths.cache_dir}")
    
    if cfg.datamodule.get("debug", False):
        logger.info("DEBUG MODE: Using a dummy datamodule with limited data.")
        dummy_data = torch.randn(100, 10)
        dummy_labels = torch.zeros(100)
        dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
        dataloader = DataLoader(
            dataset, 
            batch_size=cfg.datamodule.batch_size, 
            num_workers=cfg.datamodule.num_workers
        )
        return dataloader
 
    if "datamodule" in cfg and cfg.datamodule is not None:
        dm = hydra.utils.instantiate(cfg.datamodule)
        dm.setup()
        return dm
    
    if "dataloader" in cfg and cfg.dataloader is not None:
        return hydra.utils.instantiate(cfg.dataloader)
    
    raise ValueError("No valid 'datamodule' or 'dataloader' found in the config.")

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

    # Instantiate embedding algorithm if specified
    if cfg.algorithm is not None:
        logger.info(f"Instantiating embedding algorithm: {cfg.algorithm._target_.split('.')[-1]}")
        algorithm = hydra.utils.instantiate(cfg.algorithm)
        
        if datamodule is None:
            raise ValueError("DataModule must be provided for embedding.")

        # Collect data from datamodule's train dataloader
        data_loader = datamodule.train_dataloader()
        data = []
        for batch in data_loader:
            inputs, _ = batch  # Assuming batch is (inputs, targets)
            data.append(inputs.cpu().numpy())
        data_np = np.concatenate(data, axis=0)
        logger.debug(f"Data shape for embedding: {data_np.shape}")

        # Perform embedding
        embeddings_result = algorithm.fit_transform(data_np)
        logger.info(f"Embedding {cfg.algorithm._target_.split('.')[-1].upper()} completed with shape: {embeddings_result.shape}")

    # Instantiate network if specified
    if cfg.network is not None:
        logger.info("Instantiating Neural Network...")
        model = hydra.utils.instantiate(cfg.network)
        logger.info(f"Network instantiated: {cfg.network._target_}")

    return embeddings_result, model

def instantiate_trainer(cfg: DictConfig) -> Trainer:
    """
    Dynamically instantiate the PyTorch Lightning Trainer from the config.
    Handles callbacks and loggers if specified.
    """
    callbacks = hydra.utils.instantiate(cfg.trainer.callbacks) if "callbacks" in cfg.trainer else None
    loggers = hydra.utils.instantiate(cfg.trainer.loggers) if "loggers" in cfg.trainer else None

    return Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )


def train_model(
    cfg: DictConfig,
    trainer: Trainer,
    datamodule: Union[LightningDataModule, DataLoader],
    embeddings: Optional[np.ndarray] = None,
):
    """
    Train stage:
      - Instantiate the model to train.
      - Call trainer.fit().
      - Optionally save checkpoint.
    """
    model = instantiate_algorithm(cfg, stage="learning", embeddings=embeddings)
    trainer.fit(model, datamodule=datamodule)

    if cfg.paths.model_ckpt:
        trainer.save_checkpoint(cfg.paths.model_ckpt)


def evaluate_model(
    cfg: DictConfig,
    trainer: Trainer,
    datamodule: Union[LightningDataModule, DataLoader],
    embeddings: Optional[np.ndarray] = None,
):
    """
    Evaluation stage:
      - Instantiate (or load) the model.
      - Call trainer.test().
    """
    model = instantiate_algorithm(cfg, stage="learning", embeddings=embeddings)
    if cfg.paths.model_ckpt:
        # If you want to load from checkpoint, you could do:
        # model = YourLightningModuleClass.load_from_checkpoint(cfg.paths.model_ckpt)
        pass

    trainer.test(model, datamodule=datamodule)


def run_pipeline(
    cfg: DictConfig,
    datamodule: LightningDataModule,
    trainer: Trainer,
):
    """
    Orchestrates the entire pipeline based on the configuration.
    Executes dimensionality reduction, training, or both.
    """
    embeddings = None
    embeddings_path = cfg.paths.embeddings_file if 'embeddings_file' in cfg.paths else None

    # Determine if Dimensionality Reduction (DR) should be performed
    perform_dr = cfg.algorithm.method is not None and embeddings_path is not None

    # Determine if Network (training) should be performed
    perform_training = cfg.network is not None

    # 1. Perform Dimensionality Reduction if needed
    if perform_dr:
        logger.info("Performing Dimensionality Reduction (DR)...")
        embeddings, _ = instantiate_algorithm(cfg, datamodule=datamodule)
        if embeddings is not None and embeddings_path:
            np.save(embeddings_path, embeddings)
            logger.info(f"Embeddings saved to {embeddings_path}")

    # 2. Instantiate and train the network if specified
    if perform_training:
        _, model = instantiate_algorithm(cfg, embeddings=embeddings)
        if model is None:
            raise ValueError("Model configuration is missing for training.")

        logger.info("Starting training pipeline...")
        trainer.fit(model, datamodule=datamodule)

        # Save model checkpoint if path is specified
        if 'model_ckpt' in cfg.paths and cfg.paths.model_ckpt:
            trainer.save_checkpoint(cfg.paths.model_ckpt)
            logger.info(f"Model checkpoint saved to {cfg.paths.model_ckpt}")

    # 3. Handle Evaluation if specified
    if 'mode' in cfg and cfg.mode == "evaluate":
        if perform_training and 'model_ckpt' in cfg.paths and cfg.paths.model_ckpt:
            logger.info("Starting evaluation pipeline...")
            evaluate_model(cfg, trainer, datamodule, embeddings)
        else:
            logger.warning("Evaluation requires a trained model. Skipping evaluation.")
