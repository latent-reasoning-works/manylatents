import logging
from typing import Optional, Tuple, Dict, Any, List

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
    setup_wandb,
)

logger = logging.getLogger(__name__)

register_configs()

@hydra.main(config_path="../src/configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    setup_logging(debug=cfg.debug)
    logger.info("Final Config:\n" + OmegaConf.to_yaml(cfg))

    # Setup wandb once for the entire workflow
    setup_wandb(cfg)
    
    lightning.seed_everything(cfg.seed, workers=True)
    
    # --- Data instantiation ---
    datamodule = instantiate_datamodule(cfg)
    datamodule.setup()
    
    # --- Callbacks ---
    trainer_cb_cfg   = cfg.trainer.get("callbacks", {})
    embedding_cb_cfg = cfg.get("callbacks", {}).get("embedding", {})

    lightning_cbs, embedding_cbs = instantiate_callbacks(
        trainer_cb_cfg,
        embedding_cb_cfg
    )
    
    if not embedding_cbs:
        logger.info("No embedding callbacks configured; skip embedding‐level hooks.")

    # --- Trainer ---
    trainer = None
    if any("model" in step.algorithm for step in cfg.workflow.steps):
        logger.info("Workflow contains Lightning modules, instantiating trainer...")
        trainer = instantiate_trainer(
            cfg,
            lightning_callbacks=lightning_cbs,
        )
    else:
        logger.info("DR-only workflow detected, skipping trainer instantiation")
    
    logger.info("Starting the experiment pipeline...")
    
    # Execute workflow
    results = execute_workflow(cfg.workflow.steps, datamodule, trainer, embedding_cbs)
    
    logger.info("Experiment complete.")
    
    if wandb.run is not None:
        wandb.run.finish()
        
    return

def execute_workflow(
    steps: List[DictConfig],
    datamodule: Any,
    trainer: Optional[lightning.Trainer],
    embedding_cbs: List[Any],
) -> Dict[str, Any]:
    """Execute the workflow steps in sequence, maintaining state between steps."""
    current_input = None
    results = {}
    
    for i, step_cfg in enumerate(steps):
        step_name = step_cfg.name
        logger.info(f"\n{'='*50}")
        logger.info(f"Executing workflow step {i+1}/{len(steps)}: {step_name}")
        logger.info(f"{'='*50}")
        
        if current_input is None:
            logger.info("Using raw data from datamodule as input")
        else:
            logger.info(f"Using embeddings from previous step as input (shape: {current_input.shape})")
        
        step_output = workflow_step(step_cfg, datamodule, trainer, current_input)
        
        if step_output and "embeddings" in step_output:
            current_input = step_output["embeddings"]
            logger.info(f"Step {step_name} produced embeddings with shape: {current_input.shape}")
            
            if "scores" in step_output:
                logger.info(f"Step {step_name} evaluation scores:")
                for metric, value in step_output["scores"].items():
                    logger.info(f"  - {metric}: {value:.4f}")
        else:
            logger.warning(f"Step {step_name} did not produce embeddings")
        
        results[step_name] = step_output
        
        # Process results through callbacks
        if step_output and embedding_cbs:
            logger.info(f"Processing {len(embedding_cbs)} callbacks for step {step_name}")
            for cb in embedding_cbs:
                cb.on_dr_end(dataset=datamodule.test_dataset, embeddings=step_output)
    
    logger.info("\nWorkflow execution complete. Final results:")
    for step_name, output in results.items():
        if output and "embeddings" in output:
            logger.info(f"- {step_name}: embeddings shape {output['embeddings'].shape}")
            if "scores" in output:
                logger.info(f"  Scores: {output['scores']}")
                
    return results

def workflow_step(
    step_cfg: DictConfig,
    datamodule: Any,
    trainer: Optional[lightning.Trainer] = None,
    input_data: Optional[torch.Tensor] = None,
    step_name: str = None,
) -> Dict[str, Any]:
    """Helper function to execute a single step in the workflow pipeline.
    
    This function handles both dimensionality reduction and Lightning model steps,
    maintaining the workflow's state and data flow between steps.
    """
    name = step_cfg.name
    algorithm_cfg = step_cfg.algorithm
    evaluation_cfg = step_cfg.get("evaluation", {})
    
    # Instantiate algorithm
    dr_module = None
    lightning_module = None
    
    if "dimensionality_reduction" in algorithm_cfg:
        dr_cfg = algorithm_cfg.dimensionality_reduction
        if "_target_" not in dr_cfg:
            raise ValueError(f"Missing _target_ in dimensionality_reduction config for step {name}")
        logger.info(f"Instantiating Dimensionality Reduction for step {name}: {dr_cfg._target_.split('.')[-1]}")
        dr_module = hydra.utils.instantiate(dr_cfg)
        
        # Execute DR algorithm
        if input_data is None:
            train_loader = datamodule.train_dataloader()
            test_loader = datamodule.test_dataloader()
            field_index, _ = determine_data_source(train_loader)
            train_tensor = torch.cat([b[field_index].cpu() for b in train_loader], dim=0)
            test_tensor = torch.cat([b[field_index].cpu() for b in test_loader], dim=0)
        else:
            train_tensor = input_data
            test_tensor = input_data

        # Use fast_dev_run_dr if enabled
        if dr_cfg.get("fast_dev_run_dr", False):
            n_samples = dr_cfg.get("n_samples_fast_dev", 100)
            logger.info(f"Using fast_dev_run_dr: limiting data to {n_samples} samples")
            train_tensor = train_tensor[:n_samples]
            test_tensor = test_tensor[:n_samples]

        logger.info(f"Running DR step {name} on data shape: {train_tensor.shape}")
        dr_module.fit(train_tensor)
        embeddings = dr_module.transform(test_tensor)
        
        outputs = {
            "embeddings": embeddings,
            "label": getattr(datamodule.test_dataset, "get_labels", lambda: None)(),
            "metadata": {"source": f"DR_{name}", "data_shape": test_tensor.shape},
        }
        
    elif "model" in algorithm_cfg:
        if trainer is None:
            raise ValueError(f"Trainer required for Lightning module in step {name} but none was provided")
            
        model_cfg = algorithm_cfg.model
        lightning_module = hydra.utils.instantiate(model_cfg, datamodule=datamodule)
        if not isinstance(lightning_module, LightningModule):
            raise TypeError(f"Model must be a LightningModule, got {type(lightning_module)}")
            
        logger.info(f"Running Lightning module step {name}")
        trainer.fit(lightning_module, datamodule=datamodule)
        
        if hasattr(lightning_module, "encode"):
            test_loader = datamodule.test_dataloader()
            test_tensor = torch.cat([b["data"].cpu() for b in test_loader], dim=0)
            latents = lightning_module.encode(test_tensor)
            latents = latents.detach().cpu().numpy() if isinstance(latents, torch.Tensor) else latents
            
            outputs = {
                "embeddings": latents,
                "label": datamodule.test_dataset.get_labels() if hasattr(datamodule.test_dataset, "get_labels") else None,
                "metadata": {"source": f"latent_{name}", "data_shape": test_tensor.shape},
            }
        else:
            outputs = None
            
    # Evaluate if enabled
    if evaluation_cfg.get("enabled", True) and outputs:
        logger.info(f"Evaluating step {name}")
        outputs["scores"] = evaluate(
            outputs,
            cfg=evaluation_cfg,
            datamodule=datamodule,
            module=dr_module or lightning_module
        )
        
    return outputs

if __name__ == "__main__":
    main()
