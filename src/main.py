import logging
from typing import Optional, Tuple

import hydra
import lightning
import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

import wandb
from src.algorithms.dimensionality_reduction import DimensionalityReductionModule
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
    if not cfg.eval_only:
        dr_module, lightning_module = instantiate_algorithm(cfg, datamodule)
    else:
        dr_module, lightning_module = None, None

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
        
    # --- DR Embedding Computation ---
    embeddings: dict = {}
    if cfg.eval_only:
        logger.info("Evaluation-only mode (DR): Loading precomputed DR outputs.")
        embeddings = load_precomputed_embeddings(cfg)
    elif "dimensionality_reduction" in cfg.algorithm and cfg.algorithm.dimensionality_reduction is not None:
        # Unroll train and test dataloaders to obtain tensors.
        ### To be replaced with a more efficient method.
        train_tensor = torch.cat([b[field_index].cpu() for b in train_loader], dim=0)
        test_tensor  = torch.cat([b[field_index].cpu() for b in test_loader],  dim=0)
        logger.info(
            f"Running Dimensionality Reduction (DR) {data_source}:\n"
            f"Train tensor shape: {train_tensor.shape}\n"
            f"Test tensor shape: {test_tensor.shape}"
        )
        ## fit, transform ops
        dr_module.fit(train_tensor)
        _embeddings = dr_module.transform(test_tensor)
        logger.info(f"DR embedding completed with shape: {_embeddings.shape}")

        label_col = cfg.get('callbacks', {}).get('embedding', {}).get('plot_embeddings', {}).get('label_col')
        if label_col is not None:
            labels = getattr(datamodule.test_dataset, "get_labels", lambda: None)(label_col)
        else:
            labels = getattr(datamodule.test_dataset, "get_labels", lambda: None)()
        
        embeddings = {## conforms to EmbeddingOutputs interface
            "embeddings": _embeddings,
            "label": labels,
            "metadata": {"source": "DR", "data_shape": test_tensor.shape},
        }
        logger.info("Evaluating embeddings (from DR output)...")
        embeddings["scores"] = evaluate(
            embeddings,
            cfg=cfg,
            datamodule=datamodule,
            module=dr_module
        ) 
        
    # --- Neural Network (Lightning) setup and evaluation ---
    model_metrics, model_error = None, None
    if cfg.algorithm.get("model"):
        if cfg.eval_only and cfg.pretrained_ckpt: ## load from checkpoint if supplied
            logger.info(f"Loading pretrained model from {cfg.pretrained_ckpt}")
            lightning_module = LightningModule.load_from_checkpoint(cfg.pretrained_ckpt)
        else: ## else instantiate a new model from its config
            lightning_module = hydra.utils.instantiate(
                cfg.algorithm.model, datamodule=datamodule
                )
                
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
    latent_embeddings: dict = {}
    if lightning_module and hasattr(lightning_module, "encode"):
        logger.info("Extracting latent embeddings using the network's encoder...")
        test_tensor = torch.cat([b["data"].cpu() for b in test_loader], dim=0)
        latents = lightning_module.encode(test_tensor)
        latents = latents.detach().cpu().numpy() if isinstance(latents, torch.Tensor) else latents
            
        logger.info(f"Latent embeddings shape: {latents.shape}")
        latent_embeddings = {## conforms to EmbeddingOutputs interface
            "embeddings": latents,
            "label": datamodule.test_dataset.get_labels() if hasattr(datamodule.test_dataset, "get_labels") else None,
            "metadata": {"source": "latent", "data_shape": test_tensor.shape},
        }
        
        logger.info("Evaluating embeddings (from encoder output)...")
        latent_embeddings["scores"] = evaluate(
            latent_embeddings,
            cfg=cfg,
            datamodule=datamodule,
        )
    # --- merge embeddings scores from DR and NN, if available ---
    dr_scores     = embeddings.get("scores", {})
    latent_scores = latent_embeddings.get("scores", {})
    embedding_metrics = {**dr_scores, **latent_scores}

    for tag, embed_dict in (("dr", embeddings), ("latent", latent_embeddings)):
        if not embed_dict:
            continue
        outputs = dict(embed_dict)
        outputs["metrics"] = embedding_metrics
        for cb in embedding_cbs:
            cb.on_dr_end(dataset=datamodule.test_dataset, embeddings=outputs)

    logger.info("Experiment complete.")
    
    if wandb.run:
        wandb.finish()
        
    return

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
