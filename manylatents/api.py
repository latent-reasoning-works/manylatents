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
        # Separate None values since Hydra can't handle them in override strings
        override_list = []
        none_keys = []

        for key, value in overrides.items():
            if value is None:
                # Track None values to set them after config composition
                # Hydra's override parser can't handle "key=None" strings
                none_keys.append(key)
                continue
            elif isinstance(value, (dict, DictConfig)):
                # For nested configs (dict or DictConfig), we need to use OmegaConf
                # This is handled by directly merging later
                continue
            elif isinstance(value, list):
                # For list values (like pipeline), handle specially
                continue
            else:
                override_list.append(f"{key}={value}")

        # Compose the base configuration
        cfg = compose(config_name="config", overrides=override_list)

        # Allow flexible field additions (disable struct mode)
        # This enables the API to accept arbitrary config overrides without schema violations
        OmegaConf.set_struct(cfg, False)

        # Handle None values by setting them directly in the config
        # This avoids Hydra's override parser which can't handle None
        for key in none_keys:
            OmegaConf.update(cfg, key, None, merge=False)

        # Merge complex overrides (dicts and lists) directly
        for key, value in overrides.items():
            if isinstance(value, (dict, list, DictConfig)):
                # Convert DictConfig to plain dict for consistent handling
                if isinstance(value, DictConfig):
                    value = OmegaConf.to_container(value, resolve=True, throw_on_missing=False)

                # If the field is None, set it directly instead of merging
                if OmegaConf.select(cfg, key) is None:
                    OmegaConf.update(cfg, key, value, merge=False)
                else:
                    OmegaConf.update(cfg, key, value, merge=True)

    logger.info("API Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Handle input_data if provided
    # We need to pass this separately because OmegaConf can't serialize numpy arrays
    input_data_holder = {'data': input_data} if input_data is not None else None

    if input_data is not None:
        logger.info(f"Received input_data with shape: {input_data.shape}. Configuring PrecomputedDataModule.")

        # Dynamically create the config for the precomputed datamodule
        # WARNING: This manual dict construction is fragile. If PrecomputedDataModule's
        # signature changes (new required params, different defaults), this code won't
        # catch it until runtime. A structured config approach would be more robust,
        # but requires refactoring the config system to use dataclasses.
        # TODO: Add integration test that validates input_data parameter handling
        # TODO: Consider migrating to structured configs (see manylatents/configs/)
        # NOTE: We don't include 'data' in the config because OmegaConf can't serialize numpy arrays
        data_cfg = {
            '_target_': 'manylatents.data.precomputed_datamodule.PrecomputedDataModule',
            'batch_size': cfg.data.get('batch_size', 128) if cfg.data else 128,
            'num_workers': cfg.data.get('num_workers', 0) if cfg.data else 0,
            'seed': cfg.seed,
        }
        cfg.data = OmegaConf.create(data_cfg)

    # Smart routing: mirror main.py logic
    is_pipeline = hasattr(cfg, 'pipeline') and cfg.pipeline is not None and len(cfg.pipeline) > 0

    if is_pipeline:
        # Route to pipeline engine
        logger.info("API detected pipeline configuration. Routing to run_pipeline()...")
        return run_pipeline(cfg, input_data_holder=input_data_holder)
    else:
        # Route to single algorithm engine
        logger.info("API detected single algorithm configuration. Routing to run_algorithm()...")
        return run_algorithm(cfg, input_data_holder=input_data_holder)
