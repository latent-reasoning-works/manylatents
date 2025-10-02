"""
API for programmatic access to manylatents.

This module provides a simple, agent-friendly interface for running
dimensionality reduction and embedding tasks programmatically, supporting
in-memory chaining of transformations.
"""

import logging
from typing import Any, Dict, Optional

import torch
import lightning
from omegaconf import DictConfig, OmegaConf
from lightning import LightningDataModule

from manylatents.experiment import (
    instantiate_datamodule,
    instantiate_algorithm,
    instantiate_callbacks,
    instantiate_trainer,
    evaluate,
)
from manylatents.main import execute_step
from manylatents.algorithms.latent_module_base import LatentModule
from manylatents.utils.data import determine_data_source

logger = logging.getLogger(__name__)


def run(
    input_data: Optional[torch.Tensor] = None,
    **overrides
) -> Dict[str, Any]:
    """
    Programmatic entry point for manylatents.

    This function allows you to run dimensionality reduction algorithms
    programmatically, either on provided data or using configured dataloaders.
    It supports in-memory chaining by accepting input_data from previous steps.

    Args:
        input_data: Optional input tensor. If provided, this data will be used
                   instead of loading from the configured datamodule.
        **overrides: Configuration overrides as keyword arguments. These should
                    follow Hydra's override syntax (e.g., algorithm.latent._target_=...).

    Returns:
        Dictionary with keys:
            - embeddings: The computed embeddings (numpy array or tensor)
            - label: Labels from the dataset (if available)
            - metadata: Dictionary with run metadata
            - scores: Evaluation metrics

    Example:
        >>> # Single run
        >>> result = run(algorithm={'latent': {'_target_': 'sklearn.decomposition.PCA', 'n_components': 10}})
        >>> embeddings = result['embeddings']
        >>>
        >>> # Chained run
        >>> result1 = run(algorithm={'latent': {'_target_': 'sklearn.decomposition.PCA', 'n_components': 50}})
        >>> result2 = run(input_data=torch.from_numpy(result1['embeddings']),
        ...               algorithm={'latent': {'_target_': 'sklearn.manifold.TSNE', 'n_components': 2}})
    """
    # Build configuration from overrides
    from hydra import compose, initialize_config_dir
    from pathlib import Path

    # Get the config directory path
    config_dir = Path(__file__).parent / "configs"
    config_dir = str(config_dir.resolve())

    # Initialize Hydra with the config directory
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        # Convert overrides dict to list of strings
        override_list = []
        for key, value in overrides.items():
            if isinstance(value, dict):
                # Handle nested configs
                for subkey, subvalue in value.items():
                    override_list.append(f"{key}.{subkey}={subvalue}")
            else:
                override_list.append(f"{key}={value}")

        # Compose the configuration
        cfg = compose(config_name="config", overrides=override_list)

    logger.info("API Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Seed for reproducibility
    lightning.seed_everything(cfg.seed, workers=True)

    # --- Setup ---
    datamodule = instantiate_datamodule(cfg)
    datamodule.setup()

    # Determine which algorithm to instantiate
    if hasattr(cfg.algorithms, 'latent') and cfg.algorithms.latent is not None:
        algorithm = instantiate_algorithm(cfg.algorithms.latent, datamodule)
    elif hasattr(cfg.algorithms, 'lightning') and cfg.algorithms.lightning is not None:
        algorithm = instantiate_algorithm(cfg.algorithms.lightning, datamodule)
    else:
        raise ValueError("No algorithm specified in configuration")

    # Callbacks
    trainer_cb_cfg = cfg.trainer.get("callbacks", {})
    embedding_cb_cfg = cfg.get("callbacks", {}).get("embedding", {})
    lightning_cbs, embedding_cbs = instantiate_callbacks(trainer_cb_cfg, embedding_cb_cfg)

    # Trainer (without loggers for API mode)
    trainer = instantiate_trainer(cfg, lightning_callbacks=lightning_cbs, loggers=[])

    # --- Data preparation ---
    if input_data is not None:
        # Use provided input data
        logger.info(f"Using provided input_data with shape: {input_data.shape}")
        train_tensor = input_data
        test_tensor = input_data
    else:
        # Load from datamodule
        train_loader = datamodule.train_dataloader()
        test_loader = datamodule.test_dataloader()
        field_index, data_source = determine_data_source(train_loader)

        train_tensor = torch.cat([b[field_index].cpu() for b in train_loader], dim=0)
        test_tensor = torch.cat([b[field_index].cpu() for b in test_loader], dim=0)

        logger.info(
            f"Loaded data from {data_source}:\n"
            f"Train shape: {train_tensor.shape}, Test shape: {test_tensor.shape}"
        )

    # --- Execute ---
    latents = execute_step(
        algorithm=algorithm,
        train_tensor=train_tensor,
        test_tensor=test_tensor,
        trainer=trainer,
        cfg=cfg,
        datamodule=datamodule
    )

    # --- Package results ---
    embeddings: Dict[str, Any] = {}

    if latents is not None:
        embeddings = {
            "embeddings": latents,
            "label": getattr(getattr(datamodule, "test_dataset", None), "get_labels", lambda: None)(),
            "metadata": {
                "source": "api",
                "algorithm_type": type(algorithm).__name__,
                "input_shape": test_tensor.shape,
                "output_shape": latents.shape if hasattr(latents, 'shape') else None,
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

    # Callback processing
    if embeddings and embedding_cbs:
        for cb in embedding_cbs:
            cb.on_latent_end(dataset=datamodule.test_dataset, embeddings=embeddings)

    logger.info("API execution complete.")
    return embeddings
