"""
Programmatic API for agent-driven workflows.

This module provides a Python function interface for manyAgents to call
manyLatents directly without subprocess overhead. It mirrors the smart
routing logic from main.py but builds configurations programmatically.

Example:
    # Single algorithm
    result = run(
        data='swissroll',
        algorithms={'latent': {'_target_': 'manylatents.algorithms.latent.pca.PCA', 'n_components': 10}}
    )

    # Pipeline (in-memory chaining)
    result = run(
        pipeline=[
            {'name': 'pca_step', 'overrides': {'algorithms': {'latent': {'n_components': 50}}}},
            {'name': 'phate_step', 'overrides': {'algorithms': {'latent': {'_target_': '...PHATE'}}}}
        ]
    )

    # Chaining with input_data
    result1 = run(data='swissroll', algorithms={'latent': 'pca'})
    result2 = run(input_data=result1['embeddings'], algorithms={'latent': 'phate'})
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# IMPORTANT: Import configs to register base_config with Hydra ConfigStore
import manylatents.configs
from manylatents.experiment import run_algorithm, run_pipeline

logger = logging.getLogger(__name__)


def run(
    input_data: Optional[np.ndarray] = None,
    **overrides: Any
) -> Dict[str, Any]:
    """
    Programmatic entry point for manyLatents.

    This function provides a clean API for agents to run dimensionality reduction
    algorithms either as single runs or sequential pipelines, with optional
    in-memory data passing.

    Args:
        input_data: Optional input tensor. If provided, this data will be used
                   instead of loading from the configured datamodule. Useful for
                   chaining multiple API calls.
        **overrides: Configuration overrides as keyword arguments. These should
                    follow Hydra's structure (e.g., data='swissroll',
                    algorithms={'latent': {...}}, pipeline=[...]).

    Returns:
        Dictionary with keys:
            - embeddings: The computed embeddings (numpy array)
            - label: Labels from the dataset (if available)
            - metadata: Dictionary with run metadata
            - scores: Evaluation metrics

    Examples:
        >>> # Single run
        >>> result = run(data='swissroll', algorithms={'latent': {'_target_': '...PCA', 'n_components': 10}})
        >>> embeddings = result['embeddings']
        >>>
        >>> # Chained runs (manual pipeline)
        >>> result1 = run(data='swissroll', algorithms={'latent': 'pca'})
        >>> result2 = run(input_data=result1['embeddings'], algorithms={'latent': 'phate'})
        >>>
        >>> # Automatic pipeline (config-driven)
        >>> result = run(pipeline=[
        ...     {'name': 'step1', 'overrides': {'algorithms': {'latent': 'pca'}}},
        ...     {'name': 'step2', 'overrides': {'algorithms': {'latent': 'phate'}}}
        ... ])
    """
    # Build configuration from overrides
    config_dir = Path(__file__).parent / "configs"
    config_dir = str(config_dir.resolve())

    # Initialize Hydra with the config directory
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        # Convert overrides dict to list of strings for Hydra
        override_list = []
        for key, value in overrides.items():
            if isinstance(value, dict):
                # For nested configs, we need to use OmegaConf
                # This is handled by directly merging later
                continue
            elif isinstance(value, list):
                # For list values (like pipeline), handle specially
                continue
            else:
                override_list.append(f"{key}={value}")

        # Compose the base configuration
        cfg = compose(config_name="config", overrides=override_list)

        # Merge complex overrides (dicts and lists) directly
        for key, value in overrides.items():
            if isinstance(value, (dict, list)):
                OmegaConf.update(cfg, key, value, merge=True)

    logger.info("API Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Handle input_data if provided
    # TODO: Implement PrecomputedDataModule for input_data support
    # For now, input_data is not supported in this version
    if input_data is not None:
        logger.warning(
            "input_data parameter is not yet supported in this version. "
            "Use pipeline configurations for sequential workflows instead."
        )

    # Smart routing: mirror main.py logic
    is_pipeline = hasattr(cfg, 'pipeline') and cfg.pipeline is not None and len(cfg.pipeline) > 0

    if is_pipeline:
        # Route to pipeline engine
        logger.info("API detected pipeline configuration. Routing to run_pipeline()...")
        return run_pipeline(cfg)
    else:
        # Route to single algorithm engine
        logger.info("API detected single algorithm configuration. Routing to run_algorithm()...")
        return run_algorithm(cfg)
