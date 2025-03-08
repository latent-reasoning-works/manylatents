import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Config:
    """Configuration schema for the experiment.

    This class defines the structure of the Hydra configuration for the experiment.
    """

    algorithm: Any
    """Configuration for the algorithm (a LightningModule)."""

    datamodule: Optional[Any] = None
    """Configuration for the datamodule (dataset + transforms + dataloader creation)."""

    callbacks: Optional[Any] = None
    """Configuration for callbacks used during training and evaluation."""

    trainer: Dict[str, Any] = field(default_factory=dict)
    """Trainer configuration (arguments passed to Lightning Trainer)."""

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

    name: str = "default"
    """Experiment name."""
