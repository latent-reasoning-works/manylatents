import logging
from typing import Optional, Tuple

import hydra
import numpy as np
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf

import src  # noqa: F401
from src.algorithms.dimensionality_reduction import DimensionalityReductionModule
from src.configs import register_configs
from src.experiment import (
    evaluate,
    instantiate_datamodule,
    instantiate_trainer,
    train_model,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

register_configs()

@hydra.main(config_path="../src/configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main entry point:
      - Instantiates datamodule, algorithm, and trainer (if needed)
      - Runs Dimensionality Reduction (DR) if specified and obtains embeddings
      - Runs training and evaluation only if a neural network is provided
      
    Note: For network-only configurations, embeddings remain None at instantiation 
    and should be extracted later from the trained model.
    """
    logger.info("Final Config:\n" + OmegaConf.to_yaml(cfg))
    logger.info("Starting the experiment pipeline...")

    # Instantiate the datamodule
    logger.info("Instantiating the datamodule...")
    datamodule = instantiate_datamodule(cfg)
    logger.info(f"Datamodule instance: {datamodule} (type: {type(datamodule)})")

    dr_module, embeddings, model = instantiate_algorithm(cfg, datamodule=datamodule)

    # Run DR if configured
    if "dimensionality_reduction" in cfg.algorithm and cfg.algorithm.dimensionality_reduction is not None:
        logger.info("Running Dimensionality Reduction (DR)...")
        logger.info(f"DR completed. Embedding shape: {embeddings.shape if embeddings is not None else 'N/A'}")
        
        dr_metric_name, dr_error, dr_metrics = evaluate(
            dr_module,
            cfg=cfg,
            datamodule=datamodule,
            embeddings=embeddings,
            
        )
    else:
        logger.info("No DR algorithm specified. Proceeding with raw/precomputed data.")

    # Run training and evaluation if a neural network is configured
    if "network" in cfg.algorithm and cfg.algorithm.network is not None:
        logger.info("Instantiating Neural Network model...")
        logger.info("Instantiating the trainer...")
        trainer = instantiate_trainer(cfg)
        logger.info("Running training...")
        train_model(cfg, trainer, datamodule, model, embeddings)
        
        model_metric_name, model_error, model_metrics = evaluate(
            model,
            cfg=cfg,
            datamodule=datamodule,
            model=model,
            embeddings=embeddings,
        )
        logger.info(f"Model evaluation completed. Error: {model_error}, Metrics: {model_metrics}")
    else:
        logger.info("No neural network specified. Skipping training and evaluation.")

    logger.info("Experiment complete.")
    
def instantiate_algorithm(
    cfg: DictConfig,
    datamodule: Optional[LightningDataModule] = None,
) -> Tuple[Optional[DimensionalityReductionModule], Optional[np.ndarray], Optional[torch.nn.Module]]:
    """
    Instantiates the algorithms specified in the configuration.
    
    The function returns a tuple: (dr module, embeddings, lightning module).
    
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
    dr_algorithm: Optional[torch.nn.Module] = None
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
                logger.debug("Calling on_dr_end() to save embeddings...")
                dr_callback.on_dr_end(embeddings_result)
            else:
                logger.warning("Callback has no on_dr_end(). Skipping.")

    if "network" in cfg.algorithm and cfg.algorithm.network is not None:
        model_cfg = cfg.algorithm.network
        if "_target_" not in model_cfg:
            raise ValueError("Missing _target_ in network config")

        logger.info(f"Instantiating Neural Network: {model_cfg._target_}")
        model = hydra.utils.instantiate(model_cfg)

    return dr_algorithm, embeddings_result, model

if __name__ == "__main__":
    main()
