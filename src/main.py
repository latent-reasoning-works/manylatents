import logging
import os
from typing import Optional, Tuple

import hydra
import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

import wandb
from src.algorithms.dimensionality_reduction import DimensionalityReductionModule
from src.configs import register_configs
from src.experiment import (
    evaluate,
    instantiate_datamodule,
    instantiate_trainer,
    train_model,
)
from src.utils.data import determine_data_source
from src.utils.utils import aggregate_metrics, setup_logging

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
        
        if wandb.run:
            wandb.run.config.update({k: v for k, v in os.environ.items() if k.startswith("SLURM")})
            ## log slurm variables; check if useful, usable across cluster envs

    datamodule = instantiate_datamodule(cfg)
    dr_module, lightning_module = instantiate_algorithm(cfg)
    dr_metrics, dr_scores, model_metrics, model_error = None, None, None, None

    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()

    field_index, data_source = determine_data_source(train_loader)

    logger.info("Starting the experiment pipeline...")

    if "dimensionality_reduction" in cfg.algorithm and cfg.algorithm.dimensionality_reduction is not None:
        if datamodule is None:
            raise ValueError("DataModule must be provided for Dimensionality Reduction.")
        
        ## Unroll train and test dataloaders to fit DR module
        train_batches = [batch[field_index].cpu() for batch in train_loader]
        train_tensor = torch.cat(train_batches, dim=0)
        
        test_batches  = [batch[field_index].cpu() for batch in test_loader]
        test_tensor = torch.cat(test_batches, dim=0)

        logger.info(
            f"Running Dimensionality Reduction (DR) on {data_source}:\n"
            f"Train tensor shape: {train_tensor.shape}\n"
            f"Test tensor shape: {test_tensor.shape}"
        )
        
        dr_module.fit(train_tensor)
        dr_embedding = dr_module.transform(test_tensor)
        
        logger.info(f"Embedding completed with shape: {dr_embedding.shape}")
        
        dr_metrics = evaluate(
            dr_module,
            cfg=cfg,
            datamodule=datamodule,
            embeddings=dr_embedding,
        )

        dr_scores = dr_metrics[next(iter(dr_metrics))] if dr_metrics else None
        
        logger.info(f"DR evaluation completed. Metrics {dr_metrics}, Scores: {dr_scores}")
        
        callback_outputs = []

        #TODO: additionalmetrics was phased out,
        #update so it's able to plot and save lightning embeddings as well
        # i.e. "integrate" both pipeline steps
        if "dimensionality_reduction" in cfg.callbacks:
            dr_callbacks = cfg.callbacks.dimensionality_reduction
            for name, cb_cfg in dr_callbacks.items():
                dr_callback = hydra.utils.instantiate(cb_cfg)
                if hasattr(dr_callback, "on_dr_end") and callable(dr_callback.on_dr_end):
                    # pass dataset and embeddings to the callback,
                    # dataset specific label, plotting logic is handled by the callback
                    output = dr_callback.on_dr_end(
                        dataset=datamodule.train_dataset, 
                        embeddings=dr_embedding,
                        )
                    callback_outputs.append((name, output))
                else:
                    logger.warning("Callback has no on_dr_end() method. Skipping metrics.")
                        
    if "network" in cfg.algorithm and cfg.algorithm.network is not None:
        logger.info("Instantiating the Neural Network model...")
        trainer = instantiate_trainer(cfg)
        logger.info("Running training...")
        train_model(cfg, trainer, datamodule, lightning_module, dr_embedding)
        
        ## verify correctness, module is being called twice
        model_metrics, model_error = evaluate(
            lightning_module,
            cfg=cfg,
            datamodule=datamodule,
            model=lightning_module,
            embeddings=dr_embedding,
        )
        logger.info(f"Model evaluation completed. Error: {model_error}, Metrics: {model_metrics}")

    aggregated_metrics = {}

    aggregated_metrics = aggregate_metrics(
        dr_metrics=dr_metrics,
        dr_scores=dr_scores,
        model_metrics=model_metrics if model_metrics else None,
        model_error=model_error,    
        )
    
    logger.info(f"Aggregated metrics: {aggregated_metrics}")
    
    if hasattr(wandb, "run") and wandb.run is not None:
        wandb.log(aggregated_metrics)
        wandb.finish()
    else:
        logger.info("wandb.run not active; skipping wandb.log")

    
    assert aggregated_metrics is not None
    logger.info("Experiment complete.")

    return aggregated_metrics

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
