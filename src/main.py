import logging
from typing import Optional, Dict, Any, List

import hydra
import lightning
import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

import wandb
from src.algorithms.latent_module_base import LatentModule
from src.configs import register_configs
from src.experiment import (
    evaluate,
    instantiate_callbacks,
    instantiate_datamodule,
    instantiate_trainer,
)
from src.utils.data import determine_data_source
from src.utils.utils import (
    load_precomputed_embeddings,
    setup_logging,
)

logger = logging.getLogger(__name__)

register_configs()

@hydra.main(config_path="../src/configs", config_name="config", version_base=None)
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

    if cfg.debug:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project=cfg.project,
            name=cfg.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    lightning.seed_everything(cfg.seed, workers=True)
    
    # --- Data instantiation ---
    datamodule = instantiate_datamodule(cfg)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()
    field_index, data_source = determine_data_source(train_loader)
    
    # --- Algorithm modules ---
    algorithms = instantiate_algorithms(cfg, datamodule)
    
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
    
    logger.info("Starting the experiment pipeline...")
        
    # --- Latent Embedding Computation ---
    embeddings: Dict[str, Any] = {}
    
    if cfg.eval_only:
        logger.info("Evaluation-only mode: Loading precomputed latent outputs.")
        embeddings = load_precomputed_embeddings(cfg)
    else:
        # Unroll train and test dataloaders to obtain tensors
        train_tensor = torch.cat([b[field_index].cpu() for b in train_loader], dim=0)
        test_tensor  = torch.cat([b[field_index].cpu() for b in test_loader],  dim=0)
        
        logger.info(
            f"Running Latent Algorithms on {data_source}:\n"
            f"Train tensor shape: {train_tensor.shape}\n"
            f"Test tensor shape: {test_tensor.shape}"
        )
        
        # Process each algorithm
        for i, algorithm in enumerate(algorithms):
            if isinstance(algorithm, LatentModule):
                logger.info(f"Processing algorithm {i+1}/{len(algorithms)}: {type(algorithm).__name__}")
                
                # Fit and transform
                algorithm.fit(train_tensor)
                _embeddings = algorithm.transform(test_tensor)
                logger.info(f"Algorithm {type(algorithm).__name__} embedding shape: {_embeddings.shape}")
                
                # Create embedding output
                algorithm_embeddings = {
                    "embeddings": _embeddings,
                    "label": getattr(datamodule.test_dataset, "get_labels", lambda: None)(),
                    "metadata": {
                        "source": f"algorithm_{i+1}", 
                        "algorithm_type": type(algorithm).__name__,
                        "data_shape": test_tensor.shape
                    },
                }
                
                # Evaluate embeddings
                logger.info(f"Evaluating embeddings from {type(algorithm).__name__}...")
                algorithm_embeddings["scores"] = evaluate(
                    algorithm_embeddings,
                    cfg=cfg,
                    datamodule=datamodule,
                    module=algorithm
                )
                
                # Use the first algorithm's output as the main result
                if not embeddings:
                    embeddings = algorithm_embeddings
                else:
                    # Merge scores from multiple algorithms
                    embeddings["scores"].update(algorithm_embeddings["scores"])
    
    # --- Neural Network processing (if any algorithms are LightningModules) ---
    lightning_modules = [alg for alg in algorithms if isinstance(alg, LightningModule)]
    
    for lightning_module in lightning_modules:
        if cfg.eval_only and cfg.pretrained_ckpt:
            logger.info(f"Loading pretrained model from {cfg.pretrained_ckpt}")
            lightning_module = LightningModule.load_from_checkpoint(cfg.pretrained_ckpt)
        
        if not cfg.eval_only:
            logger.info("Running training...")
            trainer.fit(lightning_module, datamodule=datamodule)
        
        logger.info("Running model evaluation.")
        evaluation_result = evaluate(
            lightning_module,
            cfg=cfg,
            trainer=trainer,
            datamodule=datamodule,
        )
        
        # Handle different return types from evaluate
        if isinstance(evaluation_result, tuple):
            model_metrics, model_error = evaluation_result
        else:
            model_metrics, model_error = evaluation_result, None
            
        logger.info(f"Model evaluation completed. Summary Error: {model_error}, Metrics: {model_metrics}")
        
        # Extract latent embeddings from the network's encoder if available
        if hasattr(lightning_module, "encode"):
            logger.info("Extracting latent embeddings using the network's encoder...")
            test_tensor = torch.cat([b["data"].cpu() for b in test_loader], dim=0)
            latents = lightning_module.encode(test_tensor)
            latents = latents.detach().cpu().numpy() if isinstance(latents, torch.Tensor) else latents
                
            logger.info(f"Latent embeddings shape: {latents.shape}")
            nn_embeddings = {
                "embeddings": latents,
                "label": datamodule.test_dataset.get_labels() if hasattr(datamodule.test_dataset, "get_labels") else None,
                "metadata": {"source": "neural_network", "data_shape": test_tensor.shape},
            }
            
            logger.info("Evaluating embeddings (from encoder output)...")
            nn_embeddings["scores"] = evaluate(
                nn_embeddings,
                cfg=cfg,
                datamodule=datamodule,
            )
            
            # Merge scores
            if embeddings:
                embeddings["scores"].update(nn_embeddings["scores"])
            else:
                embeddings = nn_embeddings

    # --- Callback processing ---
    if embeddings and embedding_cbs:
        for cb in embedding_cbs:
            cb.on_latent_end(dataset=datamodule.test_dataset, embeddings=embeddings)

    logger.info("Experiment complete.")
    
    if wandb.run:
        wandb.finish()
        
    return embeddings

def instantiate_algorithms(cfg: DictConfig, datamodule) -> List[Any]:
    """Instantiate all algorithms from the algorithms list in config."""
    algorithms = []
    
    if "algorithms" not in cfg:
        logger.warning("No algorithms configured in config")
        return algorithms
    
    for i, algorithm_cfg in enumerate(cfg.algorithms):
        if "_target_" not in algorithm_cfg:
            raise ValueError(f"Missing _target_ in algorithm config {i}")
        
        logger.info(f"Instantiating Algorithm {i+1}: {algorithm_cfg._target_.split('.')[-1]}")
        
        # Check if this is a LightningModule that needs datamodule
        if "model" in algorithm_cfg._target_.lower() or "lightning" in algorithm_cfg._target_.lower():
            algorithm = hydra.utils.instantiate(algorithm_cfg, datamodule=datamodule)
        else:
            algorithm = hydra.utils.instantiate(algorithm_cfg)
            
        algorithms.append(algorithm)
    
    return algorithms

if __name__ == "__main__":
    main()
