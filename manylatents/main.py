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
from manylatents.configs import register_configs
from manylatents.experiment import (
    evaluate,
    instantiate_callbacks,
    instantiate_datamodule,
    instantiate_trainer,
)
from manylatents.utils.data import determine_data_source
from manylatents.utils.utils import (
    load_precomputed_embeddings,
    setup_logging,
)

logger = logging.getLogger(__name__)

register_configs()

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
        
    # --- Sequential Algorithm Execution ---
    embeddings: Dict[str, Any] = {}
    
    if cfg.eval_only:
        logger.info("Evaluation-only mode: Loading precomputed latent outputs.")
        embeddings = load_precomputed_embeddings(cfg)
    else:
        # Unroll train and test dataloaders to obtain tensors
        train_tensor = torch.cat([b[field_index].cpu() for b in train_loader], dim=0)
        test_tensor  = torch.cat([b[field_index].cpu() for b in test_loader],  dim=0)
        
        logger.info(
            f"Running Sequential Algorithm Workflow on {data_source}:\n"
            f"Train tensor shape: {train_tensor.shape}\n"
            f"Test tensor shape: {test_tensor.shape}\n"
            f"Workflow length: {len(algorithms)} algorithms"
        )
        
        # Execute algorithms sequentially - each algorithm's output feeds into the next
        current_data = test_tensor
        all_scores = {}
        
        for i, algorithm in enumerate(algorithms):
            logger.info(f"Executing algorithm {i+1}/{len(algorithms)}: {type(algorithm).__name__}")
            
            latents = None
            
            # --- Compute latents based on algorithm type ---
            if isinstance(algorithm, LatentModule):
                # LatentModule: fit/transform pattern
                algorithm.fit(train_tensor)  # Always fit on original train data
                latents = algorithm.transform(current_data)
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
                    latents = algorithm.encode(current_data)
                    latents = latents.detach().cpu().numpy() if isinstance(latents, torch.Tensor) else latents
                    logger.info(f"LightningModule embedding shape: {latents.shape}")
                else:
                    logger.warning(f"LightningModule {type(algorithm).__name__} has no 'encode' method - skipping")
                    continue
            
            # --- Unified embedding wrapping (same for all algorithm types) ---
            if latents is not None:
                algorithm_embeddings = {
                    "embeddings": latents,
                    "label": getattr(getattr(datamodule, "test_dataset", None), "get_labels", lambda: None)(),
                    "metadata": {
                        "source": f"algorithm_{i+1}", 
                        "algorithm_type": type(algorithm).__name__,
                        "data_shape": current_data.shape,
                        "workflow_position": i+1,
                        "workflow_total": len(algorithms)
                    },
                }
                
                # Evaluate embeddings
                logger.info(f"Evaluating embeddings from {type(algorithm).__name__}...")
                algorithm_embeddings["scores"] = evaluate(
                    algorithm_embeddings,
                    cfg=cfg,
                    datamodule=datamodule,
                    module=algorithm if isinstance(algorithm, LatentModule) else None
                )
                
                # Update scores collection
                all_scores.update(algorithm_embeddings["scores"])
                
                # Use the last algorithm's output as final embeddings
                embeddings = algorithm_embeddings
                
                # Update current_data for next algorithm in workflow
                current_data = torch.tensor(latents) if not isinstance(latents, torch.Tensor) else latents
                logger.info(f"Updated workflow data shape for next algorithm: {current_data.shape}")
        
        # Ensure final embeddings contain all scores from the workflow
        if embeddings:
            embeddings["scores"] = all_scores
            embeddings["metadata"]["workflow_final"] = True

    # --- Callback processing ---
    if embeddings and embedding_cbs:
        for cb in embedding_cbs:
            cb.on_latent_end(dataset=datamodule.test_dataset, embeddings=embeddings)

    logger.info("Experiment complete.")
    
    if wandb.run:
        wandb.finish()
        
    return embeddings

def instantiate_algorithms(cfg: DictConfig, datamodule) -> List[Any]:
    """Instantiate algorithms from configuration, supporting sequential workflows and list expansion.
    
    Supports:
    1. Sequential workflows: [pca, umap, autoencoder] - each processes previous output
    2. List parameter expansion: n_components: [2, 5, 10] creates multiple algorithm instances
    3. Parallel mode: (placeholder for future implementation)
    
    Args:
        cfg: Configuration containing algorithms specification
        datamodule: DataModule instance
        
    Returns:
        List of algorithm instances ready for sequential execution
    """
    if "algorithm" not in cfg or cfg.algorithm is None:
        logger.warning("No algorithm configured in config")
        return []
    
    # Check for parallel mode (placeholder - not implemented yet)
    execution_mode = getattr(cfg, 'algorithm_execution_mode', 'sequential')
    if execution_mode == 'parallel':
        raise NotImplementedError(
            "Parallel algorithm execution not yet implemented. "
            "This will require integration with hydra launcher for proper distributed execution."
        )
    
    # Expand algorithm configurations similar to flatten_and_unroll_metrics
    expanded_configs = _expand_algorithm_configs(cfg.algorithm)
    
    algorithms = []
    for i, algorithm_cfg in enumerate(expanded_configs):
        algorithm = _instantiate_algorithm_config(algorithm_cfg, datamodule)
        algorithms.append(algorithm)
        logger.info(f"Instantiated algorithm {i+1}/{len(expanded_configs)}: {type(algorithm).__name__}")
    
    return algorithms


def _expand_algorithm_configs(algorithm_configs) -> List[DictConfig]:
    """Expand algorithm configurations with list-valued parameters.
    
    Handles both nested dict structures (from overrides) and direct list formats.
    Similar to flatten_and_unroll_metrics, creates multiple algorithm instances
    when parameters have list values.
    
    Example inputs:
        Nested dict: {latent: {_target_: PCA, n_components: [2, 5, 10]}}
        Direct list: [{_target_: PCA, n_components: 2}]
    """
    # Handle nested dict structure from overrides (e.g. {latent: {...}, lightning: {...}})
    if isinstance(algorithm_configs, DictConfig) and not algorithm_configs.get("_target_"):
        configs_list = []
        # Extract algorithm configs from nested structure
        for algo_type in ["latent", "lightning"]:
            if algo_type in algorithm_configs and algorithm_configs[algo_type] is not None:
                configs_list.append(algorithm_configs[algo_type])
        algorithm_configs = configs_list
    
    # Ensure we have a list
    if not isinstance(algorithm_configs, (list, ListConfig)):
        algorithm_configs = [algorithm_configs]
    
    expanded = []
    
    for config in algorithm_configs:
        if not isinstance(config, DictConfig):
            expanded.append(config)
            continue
        
        # Find parameters with list values (similar to flatten_and_unroll_metrics)
        sweep_keys = []
        sweep_vals = []
        for k, v in config.items():
            if isinstance(v, (list, tuple, ListConfig)) and k != "_target_":
                sweep_keys.append(k)
                sweep_vals.append(list(v))
        
        if not sweep_keys:
            # No list parameters, use as-is
            expanded.append(config)
        else:
            # Create cartesian product of list parameters
            for combo in product(*sweep_vals):
                config_copy = copy.deepcopy(config)
                for k, val in zip(sweep_keys, combo):
                    # Convert to native types
                    if isinstance(val, ListConfig): 
                        val = list(val)
                    if isinstance(val, float) and float(val).is_integer():
                        val = int(val)
                    setattr(config_copy, k, val)
                expanded.append(config_copy)
    
    return expanded


def _instantiate_algorithm_config(algorithm_cfg: DictConfig, datamodule) -> Any:
    """Instantiate a single algorithm configuration.
    
    Args:
        algorithm_cfg: Algorithm configuration with _target_ and parameters
        datamodule: DataModule instance for algorithms that need it
        
    Returns:
        Instantiated algorithm (LatentModule or LightningModule)
    """
    if "_target_" not in algorithm_cfg:
        raise ValueError("Missing _target_ in algorithm config")
    
    target = algorithm_cfg._target_
    logger.info(f"Instantiating {target.split('.')[-1]}")
    
    # Check if this is a LightningModule that needs datamodule
    if ("model" in target.lower() or 
        "lightning" in target.lower() or 
        "networks" in target.lower()):
        return hydra.utils.instantiate(algorithm_cfg, datamodule=datamodule)
    else:
        return hydra.utils.instantiate(algorithm_cfg)

if __name__ == "__main__":
    main()
