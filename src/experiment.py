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

    if cfg.ckpt_dir:
        trainer.save_checkpoint(cfg.ckpt_dir)


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
    if cfg.ckpt_dir:
        # If you want to load from checkpoint, you could do:
        # model = YourLightningModuleClass.load_from_checkpoint(cfg.ckpt_dir)
        pass

    trainer.test(model, datamodule=datamodule)

def run_pipeline(
    cfg: DictConfig,
    datamodule: LightningDataModule,
    trainer: Trainer,
    algorithm: Optional[torch.nn.Module] = None,
    embeddings: Optional[np.ndarray] = None,
):
    """
    Orchestrates the entire pipeline:
    - Performs dimensionality reduction (if applicable)
    - Trains a model (if specified)
    - Evaluates (if specified)
    """
    perform_dr = "dimensionality_reduction" in cfg.algorithm
    perform_training = "network" in cfg.algorithm
    
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy() ## ugly, to be fixed with type checks

    if embeddings is not None and embeddings.size > 0:
        logger.info(f"Using precomputed embeddings with shape: {embeddings.shape}")

    elif perform_dr:
        logger.info("Performing Dimensionality Reduction (DR)...")
        data_loader = datamodule.train_dataloader()
        data = [inputs.cpu().numpy() for inputs, _ in data_loader]
        data_np = np.concatenate(data, axis=0)
        algorithm.fit(torch.tensor(data_np))
        embeddings = algorithm.transform(torch.tensor(data_np))
        
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        logger.info(f"Embedding completed with shape: {embeddings.shape}")

    elif perform_training and (embeddings is None or embeddings.size == 0):
        logger.info("No DR applied — using raw data for Neural Network.")
        data_loader = datamodule.train_dataloader()
        data = [inputs.cpu().numpy() for inputs, _ in data_loader]
        embeddings = np.concatenate(data, axis=0)

    if perform_training:
        logger.info("Training Neural Network...")
        train_model(cfg, trainer, datamodule, embeddings)

    if cfg.get("mode", "train") == "evaluate" and perform_training:
        evaluate_model(cfg, trainer, datamodule, embeddings)
