import logging
from typing import Optional, Tuple

import hydra
import numpy as np
import torch
import wandb
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

from src.algorithms.dimensionality_reduction import DimensionalityReductionModule
from src.configs import register_configs
from src.experiment import (
    evaluate,
    instantiate_datamodule,
    instantiate_trainer,
    train_model,
)
from src.utils.utils import setup_logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
    setup_logging(debug=cfg.debug)

    logger.info("Final Config:\n" + OmegaConf.to_yaml(cfg))
    
    if cfg.debug:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            ## add project name?
            name=cfg.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    datamodule = instantiate_datamodule(cfg)
    dr_module, lightning_module = instantiate_algorithm(cfg)
    dr_metrics = {}
    
    logger.info("Starting the experiment pipeline...")

    if "dimensionality_reduction" in cfg.algorithm and cfg.algorithm.dimensionality_reduction is not None:
        logger.info("Running Dimensionality Reduction (DR)...")
        
        if datamodule is None:
            raise ValueError("DataModule must be provided for Dimensionality Reduction.")
        data_loader = datamodule.train_dataloader()
        data_list = [inputs.cpu().numpy() for inputs, _ in data_loader]
        data_np = np.concatenate(data_list, axis=0)
        
        dr_module.fit(torch.tensor(data_np))
        dr_embedding = dr_module.transform(torch.tensor(data_np))
        
        logger.info(f"Embedding completed with shape: {dr_embedding.shape}")
        
        dr_metric_name, dr_error, dr_metrics = evaluate(
            dr_module,
            cfg=cfg,
            datamodule=datamodule,
            embeddings=dr_embedding,
        )

        if "dimensionality_reduction" in cfg.callbacks:
            dr_callbacks = cfg.callbacks.dimensionality_reduction
            for name, cb_cfg in dr_callbacks.items():
                dr_callback = hydra.utils.instantiate(cb_cfg)
                if hasattr(dr_callback, "on_dr_end") and callable(dr_callback.on_dr_end):
                # Pass the original data and embeddings to the callback.
                    X = datamodule.train_dataset.full_data
                    dr_callback.on_dr_end(X, dr_embedding)
                else:
                    logger.warning("Callback has no on_dr_end() method. Skipping metrics.")
        else:
            logger.info("No DR algorithm specified. Proceeding with raw/precomputed data.")
            
            if not cfg.debug: ## todo: interface metrics with callbacks
                wandb.log({"DR Error": dr_error, **dr_metrics})
                wandb.log({"DR Embeddings": wandb.Image(dr_embedding.numpy())})


    if "network" in cfg.algorithm and cfg.algorithm.network is not None:
        logger.info("Instantiating Neural Network model...")
        logger.info("Instantiating the trainer...")
        trainer = instantiate_trainer(cfg)
        logger.info("Running training...")
        train_model(cfg, trainer, datamodule, lightning_module, dr_embedding)
        
        ## verify correctness, module is being called twice
        model_metric_name, model_error, model_metrics = evaluate(
            lightning_module,
            cfg=cfg,
            datamodule=datamodule,
            model=lightning_module,
            embeddings=dr_embedding,
        )
        logger.info(f"Model evaluation completed. Error: {model_error}, Metrics: {model_metrics}")
    else:
        logger.info("No neural network specified. Skipping training and evaluation.")

    logger.info("Experiment complete.")

    if wandb.run:
        wandb.finish()
    
def instantiate_algorithm(
    cfg: DictConfig,
) -> Tuple[Optional[DimensionalityReductionModule], Optional[LightningModule]]:
    """
    Instantiates the algorithms specified in the configuration.    
    
    Args:
        cfg (DictConfig): Hydra configuration. UPDATE WITH ALGORITHM CONFIG
    
    Returns:
        Tuple[Optional[DimensionalityReductionModule], Optional[LightningModule]]:
            - dr_module: Instantiated DR algorithm, or None if not specified.
            - lightning_module: Instantiated neural network model, or None if only DR is used.
    """
    dr_module: Optional[DimensionalityReductionModule] = None
    lightning_module: Optional[LightningModule] = None

    # --- DR Setup ---
    if "dimensionality_reduction" in cfg.algorithm and cfg.algorithm.dimensionality_reduction is not None:
        dr_cfg = cfg.algorithm.dimensionality_reduction
        if "_target_" not in dr_cfg:
            raise ValueError("Missing _target_ in dimensionality_reduction config")

        logger.info(f"Instantiating Dimensionality Reduction: {dr_cfg._target_.split('.')[-1]}")
        dr_module = hydra.utils.instantiate(dr_cfg)
 
    # --- NN Setup ---
    if "network" in cfg.algorithm and cfg.algorithm.network is not None:
        model_cfg = cfg.algorithm.network
        if "_target_" not in model_cfg:
            raise ValueError("Missing _target_ in network config")

        logger.info(f"Instantiating Neural Network: {model_cfg._target_}")
        lightning_module = hydra.utils.instantiate(model_cfg)

    return dr_module, lightning_module

if __name__ == "__main__":
    main()
