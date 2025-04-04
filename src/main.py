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
      - Instantiates datamodule, algorithm, and trainer as needed.
      - Computes embeddings either via DR or via a neural network’s encoder.
      - Runs training and evaluation.      
      
    This version builds a unified embedding container (a dict with at least "embeddings")
    that is passed to a unified evaluate() routine (registered for dict types) to compute
    dataset-level and module-level metrics.
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

    # Data instantiation
    datamodule = instantiate_datamodule(cfg)
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()
    field_index, data_source = determine_data_source(train_loader)

    if not cfg.eval_only:
        dr_module, lightning_module = instantiate_algorithm(cfg, datamodule)
    else:
        dr_module, lightning_module = None, None
        
    logger.info("Starting the experiment pipeline...")
        
    # --- DR Embedding Computation ---
    embeddings = None
    if cfg.eval_only:
        logger.info("Evaluation-only mode (DR): Loading precomputed DR outputs.")
        embeddings = load_precomputed_embeddings(cfg)
    elif "dimensionality_reduction" in cfg.algorithm and cfg.algorithm.dimensionality_reduction is not None:
        # Unroll train and test dataloaders to obtain tensors.
        ## To be replaced with a more efficient method.
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
        _embeddings = dr_module.transform(test_tensor)
        logger.info(f"DR embedding completed with shape: {_embeddings.shape}")
        dr_labels = datamodule.test_dataset.get_labels() if hasattr(datamodule.test_dataset, "get_labels") else None
        ## conforms to EmbeddingOutputs interface
        embeddings = {
            "embeddings": _embeddings,
            "label": dr_labels,
            "metadata": {"source": "DR", "data_shape": test_tensor.shape},
        }

    if embeddings is not None:
        logger.info("Evaluating embeddings (from DR output)...")
        embeddings["scores"] = evaluate(
            embeddings,
            cfg=cfg,
            datamodule=datamodule,
        )
        
    # --- Neural Network (Lightning) setup and evaluation ---
    model_metrics, model_error = None, None
    if "model" in cfg.algorithm and cfg.algorithm.model is not None:
        if cfg.eval_only and cfg.pretrained_ckpt:
            logger.info(f"Loading pretrained model from {cfg.pretrained_ckpt}")
            lightning_module = LightningModule.load_from_checkpoint(cfg.pretrained_ckpt)
        else:
            lightning_module = hydra.utils.instantiate(cfg.algorithm.model, datamodule=datamodule)
        
        trainer = instantiate_trainer(cfg)
        
        if not cfg.eval_only:
            logger.info("Running training...")
            trainer.fit(lightning_module, datamodule=datamodule)
        
        logger.info("Running model evaluation.")
        model_metrics, model_error = evaluate(
            lightning_module,
            cfg=cfg,
            trainer=trainer,
            datamodule=datamodule,
        )
        logger.info(f"Model evaluation completed. Summary Error: {model_error}, Metrics: {model_metrics}")
        
    # --- Additional Evaluation for Latent Embeddings from the NN (if available) ---
    latent_embeddings = None    
    if lightning_module is not None and hasattr(lightning_module, "encode"):
        logger.info("Extracting latent embeddings using the network's encoder...")
        test_batches = [batch["data"].cpu() for batch in test_loader]
        test_tensor = torch.cat(test_batches, dim=0)
        _latent_embeddings = lightning_module.encode(test_tensor)
        if isinstance(_latent_embeddings, torch.Tensor):
            _latent_embeddings = _latent_embeddings.detach().cpu().numpy()
        logger.info(f"Latent embeddings shape: {_latent_embeddings.shape}")
        latent_embeddings = {
            "embeddings": _latent_embeddings,
            "metadata": {"source": "latent", "data_shape": test_tensor.shape},
        }
        
        logger.info("Evaluating embeddings (from encoder output)...")
        latent_embeddings["scores"] = evaluate(
            latent_embeddings,
            cfg=cfg,
            datamodule=datamodule,
        )

    # --- Callbacks ---
    callback_outputs = []
    if "dimensionality_reduction" in cfg.callbacks and embeddings is not None:
        dr_callbacks = cfg.callbacks.dimensionality_reduction
        for name, cb_cfg in dr_callbacks.items():
            dr_callback = hydra.utils.instantiate(cb_cfg)
            if hasattr(dr_callback, "on_dr_end") and callable(dr_callback.on_dr_end):
                output = dr_callback.on_dr_end(dataset=datamodule.test_dataset, 
                                               embeddings=embeddings)
                callback_outputs.append((name, output))
            else:
                logger.warning("Callback has no on_dr_end() method. Skipping metrics.")
    
    # --- Aggregate embedding and model metrics ---
    aggregated_metrics = aggregate_metrics(
        dr_scores=embeddings.get("scores") if embeddings is not None else None,
        latent_scores=latent_embeddings.get("scores") if latent_embeddings is not None else None,
        model_metrics=model_metrics,
        model_error=model_error,
        callback_outputs=callback_outputs,
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
