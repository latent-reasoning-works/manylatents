import logging
from typing import Optional, Tuple, Union

import hydra
import numpy as np
import torch
from lightning import LightningDataModule, Trainer
from omegaconf import DictConfig
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
    
    if cfg.algorithm is not None:
        if "_target_" in cfg.algorithm:
            logger.info(f"Instantiating algorithm: {cfg.algorithm._target_.split('.')[-1]}")
            algorithm = hydra.utils.instantiate(cfg.algorithm)
        elif "dimensionality_reduction" in cfg.algorithm:
            dr_cfg = cfg.algorithm.dimensionality_reduction
            if "_target_" not in dr_cfg:
                raise ValueError("Missing _target_ in dimensionality_reduction config")
            logger.info(f"Instantiating dimensionality reduction: {dr_cfg._target_.split('.')[-1]}")
            algorithm = hydra.utils.instantiate(dr_cfg)
        else:
            raise ValueError("Algorithm configuration is invalid.")

        if datamodule is None:
            raise ValueError("DataModule must be provided for embedding.")

        # Load data from datamodule
        data_loader = datamodule.train_dataloader()
        data = [inputs.cpu().numpy() for inputs, _ in data_loader]
        data_np = np.concatenate(data, axis=0)
        logger.debug(f"Data shape for embedding: {data_np.shape}")

        # 🚀 Separate fit and transform
        algorithm.fit(torch.tensor(data_np))
        embeddings_result = algorithm.transform(torch.tensor(data_np))
        logger.info(f"Embedding completed with shape: {embeddings_result.shape}")

    # Neural Network Instantiation
    if "network" in cfg.algorithm:
        logger.info("Instantiating Neural Network...")
        model_cfg = cfg.algorithm.network
        if "_target_" not in model_cfg:
            raise ValueError("Missing _target_ in network config")
        model = hydra.utils.instantiate(model_cfg)
        logger.info(f"Network instantiated: {model_cfg._target_}")

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

def run_pipeline(cfg: DictConfig, datamodule: LightningDataModule, trainer: Trainer):
    """
    Orchestrates the entire pipeline:
    - Performs dimensionality reduction (if applicable)
    - Trains a model (if specified)
    - Evaluates (if specified)
    """

    logger.info("Running pipeline...")
    embeddings = None
    embeddings_path = cfg.paths.embeddings_file if "embeddings_file" in cfg.paths else None

    # Check if dimensionality reduction is needed
    perform_dr = "dimensionality_reduction" in cfg.algorithm
    perform_training = "network" in cfg.algorithm

    # 1. Perform PCA or other dimensionality reduction
    if perform_dr:
        logger.info("Performing Dimensionality Reduction (DR)...")
        embeddings, _ = instantiate_algorithm(cfg, datamodule=datamodule)

        if embeddings is not None and embeddings_path:
            np.save(embeddings_path, embeddings)
            logger.info(f"Embeddings saved to {embeddings_path}")

    # 2. Train network if applicable
    if perform_training:
        _, model = instantiate_algorithm(cfg)
        if model is None:
            raise ValueError("Model configuration is missing for training.")

        train_model(cfg, trainer, datamodule, model)

    # 3. Run evaluation if specified
    if cfg.get("mode", "train") == "evaluate" and perform_training:
        evaluate_model(cfg, trainer, datamodule, model)