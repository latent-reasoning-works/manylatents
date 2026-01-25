import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List


@dataclass
class Config:
    """Configuration schema for the experiment.

    This class defines the structure of the Hydra configuration for the experiment.
    Schema for individual algorithm components is automatically derived from their 
    Python class __init__ signatures via the config groups in algorithms/latent/ 
    and algorithms/lightning/.
    """

    # Algorithms configuration - schema automatically derived from algorithm class signatures
    algorithms: Optional[Dict[str, Any]] = None
    """Configuration for algorithms - supports nested latent and lightning configurations."""

    pipeline: List[Any] = field(default_factory=list)
    """An ordered list of algorithms to run sequentially."""

    data: Optional[Any] = None
    """Configuration for the data (dataset + transforms + dataloader creation)."""

    callbacks: Optional[Any] = None
    """Configuration for callbacks used during training and evaluation."""

    logger: Optional[Any] = None
    """Configuration for experiment-level logging (e.g., wandb)."""

    trainer: Dict[str, Any] = field(default_factory=dict)
    """Trainer configuration (arguments passed to Lightning Trainer)."""

    metrics: Optional[Any] = None
    """Configuration for metrics used during training and evaluation."""
    
    log_level: str = "info"
    """Logging level (one of: "debug", "info", "warning", "error", "critical")."""

    seed: int = field(default_factory=lambda: random.randint(0, int(1e5)))
    """Random seed for reproducibility."""

    debug: bool = False
    """Enable debug mode (disables certain loggers and enables more verbose outputs)."""

    verbose: bool = False
    """Enable verbose mode for more detailed logs."""

    cache_dir: Optional[str] = "outputs/cache"
    """Directory for caching intermediate outputs."""

    ckpt_dir: Optional[str] = "outputs/ckpt"
    """Directory to store model checkpoints."""

    data_dir: str = "./data"
    """Directory for data files. Override via CLI or MANYLATENTS_DATA_DIR env var."""

    output_dir: str = "./outputs"
    """Directory for output files (embeddings, results). Override via CLI or MANYLATENTS_OUTPUT_DIR env var."""

    name: str = "default"
    """Experiment name."""
    
    project: str = "default"
    """Project name for logging and tracking."""
    
    eval_only: bool = False
    """If True, skip training and only run evaluation (or plotting) using precomputed embeddings or a pretrained network."""
    
    pretrained_ckpt: Optional[str] = None
    """If specified, load a pretrained checkpoint to compute embeddings instead of training a new model."""
