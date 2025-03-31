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
from src.utils.utils import (
    aggregate_metrics,
    load_precomputed_embeddings,
    setup_logging,
)

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

    # Data instantiatoin
    datamodule = instantiate_datamodule(cfg)
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()
    field_index, data_source = determine_data_source(train_loader)

    if not cfg.eval_only:
        dr_module, lightning_module = instantiate_algorithm(cfg, datamodule)
    else:
        dr_module, lightning_module = None, None
        
    logger.info("Starting the experiment pipeline...")

    dr_outputs = {}
    dr_metrics = None
    model_metrics, model_error = None, None

    # ----------------------------- #
    #  Dimensionality Reduction   #
    # ----------------------------- #
    if cfg.eval_only:
        logger.info("Evaluation-only mode (DR): Loading precomputed DR outputs.")
        dr_outputs = load_precomputed_embeddings(cfg)
        dr_embedding = dr_outputs.get("embeddings")
        if hasattr(datamodule.test_dataset, "get_labels"):
            dr_outputs["label"] = datamodule.test_dataset.get_labels()
    else:
        if "dimensionality_reduction" in cfg.algorithm and cfg.algorithm.dimensionality_reduction is not None:
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
       
        if dr_module is None:
            raise ValueError(
                "Dimensionality reduction module is not instantiated."
                "Please ensure that 'dimensionality_reduction' is properly configured in your algorithm config,"
                "or set 'eval_only: true' to load precomputed embeddings."
            )
            
        dr_module.fit(train_tensor)
        dr_embedding = dr_module.transform(test_tensor)
        logger.info(f"Embedding completed with shape: {dr_embedding.shape}")
        if hasattr(datamodule.test_dataset, "get_labels"):
            dr_labels = datamodule.test_dataset.get_labels()
        else:
            dr_labels = None
            
        dr_metrics = evaluate(
            dr_module,
            cfg=cfg,
            datamodule=datamodule,
            embeddings=dr_embedding,
        )

        logger.info(f"DR Metrics: {dr_metrics}")
        dr_outputs = {
            "embeddings": dr_embedding,
            "label": dr_labels,
            "scores": dr_metrics,
            "metadata": None, ## change
        }                

    # ----------------------------- #
    #         Lightning Model     #
    # ----------------------------- #
    if "model" in cfg.algorithm and cfg.algorithm.model is not None:
        # In eval-only mode, if a pretrained checkpoint is provided, load it.
        if cfg.eval_only and cfg.pretrained_ckpt:
            logger.info(f"Loading pretrained model from {cfg.pretrained_ckpt}")
            lightning_module = LightningModule.load_from_checkpoint(cfg.pretrained_ckpt)
        else:
            lightning_module = hydra.utils.instantiate(cfg.algorithm.model, datamodule=datamodule)
        
        trainer = instantiate_trainer(cfg)
        
        if not cfg.eval_only:
            logger.info("Running training...")
            train_model(cfg, trainer, datamodule, lightning_module, dr_embedding)
        
        logger.info("Running model evaluation.")
        model_metrics, model_error = evaluate(
            lightning_module,
            cfg=cfg,
            trainer=trainer,
            datamodule=datamodule,
            embeddings=dr_embedding,
        )
        if model_metrics:
            model_error = next(iter(model_metrics.values()))
        logger.info(f"Model evaluation completed. Summary Error: {model_error}, Metrics: {model_metrics}")
    else:
        lightning_module = None  
        
    # ----------------------------- #
    #         Callbacks           #
    # ----------------------------- #
    
    callback_outputs = []

    if "dimensionality_reduction" in cfg.callbacks and dr_outputs:
        dr_callbacks = cfg.callbacks.dimensionality_reduction
        for name, cb_cfg in dr_callbacks.items():
            dr_callback = hydra.utils.instantiate(cb_cfg)
            if hasattr(dr_callback, "on_dr_end") and callable(dr_callback.on_dr_end):
                output = dr_callback.on_dr_end(dataset=datamodule.test_dataset, dr_outputs=dr_outputs)
                callback_outputs.append((name, output))
            else:
                logger.warning("Callback has no on_dr_end() method. Skipping metrics.")

    aggregated_metrics = aggregate_metrics(
        dr_metrics=dr_outputs.get("scores"),
        model_metrics=model_metrics if model_metrics else None,
        model_error=model_error,
        callback_outputs=callback_outputs
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


def instantiate_algorithm(cfg: DictConfig, datamodule) -> Tuple[Optional[DimensionalityReductionModule], Optional[LightningModule]]:
    dr_module = None
    lightning_module = None

    if "dimensionality_reduction" in cfg.algorithm and cfg.algorithm.dimensionality_reduction is not None:
        dr_cfg = cfg.algorithm.dimensionality_reduction
        if "_target_" not in dr_cfg:
            raise ValueError("Missing _target_ in dimensionality_reduction config")
        logger.info(f"Instantiating Dimensionality Reduction: {dr_cfg._target_.split('.')[-1]}")
        dr_module = hydra.utils.instantiate(dr_cfg)

    if "model" in cfg.algorithm and cfg.algorithm.model is not None:
        model_cfg = cfg.algorithm.model
        # Dynamically instantiate or use the already-instantiated module.
        lightning_module = instantiate_model(model_cfg, datamodule)
        if not isinstance(lightning_module, LightningModule):
            raise TypeError(f"Model must be a LightningModule, got {type(lightning_module)}")
    
    return dr_module, lightning_module

def instantiate_model(cfg_model, datamodule):
    if isinstance(cfg_model, (dict)) or OmegaConf.is_config(cfg_model):
        model = hydra.utils.instantiate(cfg_model, datamodule=datamodule)
    else:
        model = cfg_model
    return model

if __name__ == "__main__":
    main()
