import logging
from pathlib import Path
from typing import Any, Dict, List, Union

from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)

def resolve_algorithm_calls(cfg: DictConfig) -> List[Dict[str, Any]]:
    """
    Resolves algorithm configurations for both single and pipeline modes.

    Returns:
        A list of instantiate-ready dictionaries, each with a `_target_` key.
    """
    if cfg.get("pipeline") and isinstance(cfg.pipeline, (list, ListConfig)) and len(cfg.pipeline) > 0:
        logger.info(f"Resolving pipeline with {len(cfg.pipeline)} steps.")
        return _resolve_from_pipeline(cfg)
    else:
        logger.info("Resolving in single algorithm mode.")
        return _resolve_from_single_mode(cfg)

def _load_base_config(full_id: str) -> DictConfig:
    """Loads a base algorithm config file and validates its structure."""
    try:
        kind, name = full_id.split('/', 1)
        if kind not in ["latent", "lightning"]:
            raise ValueError("Invalid kind")
    except ValueError:
        raise ValueError(
            f"Invalid algorithm ID '{full_id}'. Must be in 'latent/<name>' or 'lightning/<name>' format."
        )

    # Get the config directory relative to this module
    config_dir = Path(__file__).parent.parent / "configs"
    algo_config_path = config_dir / "algorithms" / f"{full_id}.yaml"
    
    if not algo_config_path.exists():
        raise FileNotFoundError(f"Algorithm config not found: {algo_config_path}")

    # The load function will fail if the path isn't valid, so extra checks are redundant.
    base = OmegaConf.load(algo_config_path)
    
    if not base.get("_target_"):
        raise ValueError(f"Config '{algo_config_path}' is missing the required '_target_' key.")

    return base


def _resolve_from_single_mode(cfg: DictConfig) -> List[Dict[str, Any]]:
    """Handles backward-compatible single algorithm mode."""
    choices = HydraConfig.get().runtime.choices
    kind = None
    if choices.get("algorithms/latent"):
        kind = "latent"
        name = choices["algorithms/latent"]
    elif choices.get("algorithms/lightning"):
        kind = "lightning"
        name = choices["algorithms/lightning"]
    else:
        logger.warning("No algorithm specified in single mode.")
        return []

    full_id = f"{kind}/{name}"
    base_config = _load_base_config(full_id)

    overrides = cfg.get("algorithms", {}).get(kind, {})
    final_config = OmegaConf.merge(base_config, overrides)

    return [OmegaConf.to_container(final_config, resolve=True)]

def _resolve_from_pipeline(cfg: DictConfig) -> List[Dict[str, Any]]:
    """Handles sequential pipeline mode from a list configuration."""
    resolved_steps = []
    pipeline_steps = cfg.pipeline
    if isinstance(pipeline_steps, str):
        pipeline_steps = [s.strip() for s in pipeline_steps.split(',')]

    for i, step in enumerate(pipeline_steps):
        if isinstance(step, str):
            full_id, overrides = step, {}
        elif isinstance(step, (dict, DictConfig)) and len(step) == 1:
            full_id = next(iter(step))
            overrides = step[full_id]
        else:
            raise TypeError(
                f"Pipeline step {i} is invalid. Item must be a string (e.g., 'latent/pca') "
                f"or a single-key map (e.g., {{'latent/pca': {{n_components: 2}}}}). Got: {step}"
            )

        base_config = _load_base_config(full_id)
        final_config = OmegaConf.merge(base_config, overrides)
        
        # Merge with the full cfg to make interpolations like ${data} available
        temp_cfg = OmegaConf.merge(cfg, {"algorithm": final_config})
        resolved_algorithm = temp_cfg.algorithm
        
        resolved_steps.append(OmegaConf.to_container(resolved_algorithm, resolve=True))

    return resolved_steps