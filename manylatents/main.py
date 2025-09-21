import copy
import logging
from itertools import product
from typing import Optional, Dict, Any, List

import hydra
import lightning
import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf, ListConfig

import wandb
from manylatents.algorithms.latent_module_base import LatentModule
# Config registration now happens automatically on import
import manylatents.configs  # This triggers ConfigStore registration
from manylatents.experiment import (
    evaluate,
    instantiate_callbacks,
    instantiate_datamodule,
    instantiate_trainer,
    instantiate_algorithm,
)
from manylatents.utils.data import determine_data_source
from manylatents.utils.utils import (
    load_precomputed_embeddings,
    setup_logging,
)

logger = logging.getLogger(__name__)

# Config registration now happens automatically on import

@hydra.main(config_path="../manylatents/configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> Dict[str, Any]:
    """
    Main entry point:
      - Instantiates datamodule, algorithms, and trainer as needed.
      - Computes embeddings through a unified latent module interface.
      - Runs training and evaluation.
      
    Returns:
        A dictionary with keys: embeddings, label, metadata, scores
    """
    setup_logging(debug=cfg.debug)
    logger.info("Final Config:\n" + OmegaConf.to_yaml(cfg))

    with wandb.init(
        project=cfg.project,
        name=cfg.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="disabled" if cfg.debug else "online",
    ) as run: 
        lightning.seed_everything(cfg.seed, workers=True)
    
        # --- Data instantiation ---
        datamodule = instantiate_datamodule(cfg)
        datamodule.setup()
        train_loader = datamodule.train_dataloader()
        test_loader = datamodule.test_dataloader()
        field_index, data_source = determine_data_source(train_loader)
        
        # --- Algorithm module ---
        # Determine which algorithm to instantiate based on configuration
        if hasattr(cfg.algorithms, 'latent') and cfg.algorithms.latent is not None:
            algorithm = instantiate_algorithm(cfg.algorithms.latent, datamodule)
        elif hasattr(cfg.algorithms, 'lightning') and cfg.algorithms.lightning is not None:
            algorithm = instantiate_algorithm(cfg.algorithms.lightning, datamodule)
        else:
            raise ValueError("No algorithm specified in configuration")
        
        # --- Callbacks ---
        trainer_cb_cfg   = cfg.trainer.get("callbacks", {})
        embedding_cb_cfg = cfg.get("callbacks", {}).get("embedding", {})

        lightning_cbs, embedding_cbs = instantiate_callbacks(
            trainer_cb_cfg,
            embedding_cb_cfg
        )
        
        if not embedding_cbs:
            logger.info("No embedding callbacks configured; skip embedding‐level hooks.")

        # --- Loggers ---
        loggers = []
        if not cfg.debug:
            for lg_conf in cfg.trainer.get("logger", {}).values():
                loggers.append(hydra.utils.instantiate(lg_conf))

        # --- Trainer ---
        trainer = instantiate_trainer(
            cfg,
            lightning_callbacks=lightning_cbs,
            loggers=loggers,
        )
        
        logger.info("Starting single algorithm execution...")
            
        # --- Single Algorithm Execution ---
        embeddings: Dict[str, Any] = {}
        
        if cfg.eval_only:
            logger.info("Evaluation-only mode: Loading precomputed latent outputs.")
            embeddings = load_precomputed_embeddings(cfg)
        else:
            # Unroll train and test dataloaders to obtain tensors
            train_tensor = torch.cat([b[field_index].cpu() for b in train_loader], dim=0)
            test_tensor  = torch.cat([b[field_index].cpu() for b in test_loader],  dim=0)
            
            logger.info(
                f"Running Single Algorithm on {data_source}:\n"
                f"Train tensor shape: {train_tensor.shape}\n"
                f"Test tensor shape: {test_tensor.shape}\n"
                f"Algorithm: {type(algorithm).__name__}"
            )
            
            latents = None
            
            # --- Compute latents based on algorithm type ---
            if isinstance(algorithm, LatentModule):
                # LatentModule: fit/transform pattern
                algorithm.fit(train_tensor)
                latents = algorithm.transform(test_tensor)
                logger.info(f"LatentModule embedding shape: {latents.shape}")
                
            elif isinstance(algorithm, LightningModule):
                # LightningModule: training or eval-only
        
                # Handle eval-only mode with pretrained checkpoint
                if cfg.eval_only and hasattr(cfg, 'pretrained_ckpt') and cfg.pretrained_ckpt:
                    logger.info(f"Loading pretrained model from {cfg.pretrained_ckpt}")
                    algorithm = LightningModule.load_from_checkpoint(cfg.pretrained_ckpt)
                
                # Training phase (if not eval_only)
                if not cfg.eval_only:
                    logger.info("Running training...")
                    trainer.fit(algorithm, datamodule=datamodule)
                
                # Model evaluation
                logger.info("Running model evaluation.")
                evaluation_result = evaluate(
                    algorithm,
                    cfg=cfg,
                    trainer=trainer,
                    datamodule=datamodule,
                )
                
                # Handle evaluation results
                if isinstance(evaluation_result, tuple):
                    model_metrics, model_error = evaluation_result
                else:
                    model_metrics, model_error = evaluation_result, None
                    
                logger.info(f"Model evaluation completed. Error: {model_error}, Metrics: {model_metrics}")
                
                # Extract embeddings from encoder
                if hasattr(algorithm, "encode"):
                    logger.info("Extracting embeddings using network encoder...")
                    latents = algorithm.encode(test_tensor)
                    latents = latents.detach().cpu().numpy() if isinstance(latents, torch.Tensor) else latents
                    logger.info(f"LightningModule embedding shape: {latents.shape}")
                else:
                    logger.warning(f"LightningModule {type(algorithm).__name__} has no 'encode' method - skipping")
            
            # --- Unified embedding wrapping ---
            if latents is not None:
                embeddings = {
                    "embeddings": latents,
                    "label": getattr(getattr(datamodule, "test_dataset", None), "get_labels", lambda: None)(),
                    "metadata": {
                        "source": "single_algorithm", 
                        "algorithm_type": type(algorithm).__name__,
                        "data_shape": test_tensor.shape,
                    },
                }
                
                # Evaluate embeddings
                logger.info(f"Evaluating embeddings from {type(algorithm).__name__}...")
                embeddings["scores"] = evaluate(
                    embeddings,
                    cfg=cfg,
                    datamodule=datamodule,
                    module=algorithm if isinstance(algorithm, LatentModule) else None
                )

        # --- Callback processing ---
        if embeddings and embedding_cbs:
            for cb in embedding_cbs:
                cb.on_latent_end(dataset=datamodule.test_dataset, embeddings=embeddings)

        logger.info("Experiment complete.")
        
        if wandb.run:
            wandb.finish()
            
        return embeddings

if __name__ == "__main__":
    main()