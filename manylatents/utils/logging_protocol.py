"""
Generic logging protocol for manyLatents.

Provides a unified interface that works for BOTH algorithm types:
- LatentModule (fit/transform, no Lightning trainer)
- LightningModule (neural networks with Lightning trainer)

This enables hierarchical logging from parent orchestrators like Geomancer
without forcing all algorithms through PyTorch Lightning.
"""

from typing import Any, Dict, Optional, Protocol, Union
from abc import abstractmethod
import logging

logger = logging.getLogger(__name__)


class LoggerProtocol(Protocol):
    """
    Protocol for loggers in manyLatents.
    
    This defines the minimal interface that both direct WandB loggers
    and Lightning-wrapped loggers must implement.
    """
    
    def log_metrics(
        self, 
        metrics: Dict[str, float], 
        step: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """Log metrics with optional step and prefix."""
        ...
    
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters/configuration."""
        ...


class DirectWandbLogger:
    """
    Direct logger for LatentModule algorithms.
    
    Wraps an existing wandb.Run and provides the LoggerProtocol interface
    WITHOUT requiring PyTorch Lightning. Used for classical algorithms
    like PCA, UMAP, t-SNE, etc.
    
    Args:
        wandb_run: Existing wandb.Run to wrap
        prefix: Optional prefix for all logged keys (e.g., "step_0/")
        
    Example:
        >>> # Parent creates run
        >>> wandb_run = wandb.init(project="geomancer")
        >>> 
        >>> # Child algorithm uses direct logger
        >>> logger = DirectWandbLogger(wandb_run, prefix="pca/")
        >>> logger.log_metrics({"explained_variance": 0.95})
        >>> # Logs to: pca/explained_variance
    """
    
    def __init__(self, wandb_run: Any, prefix: str = ""):
        """
        Initialize direct logger with existing WandB run.
        
        Args:
            wandb_run: Existing wandb.Run instance
            prefix: Optional prefix for all keys (auto-adds trailing slash)
        """
        self._wandb_run = wandb_run
        
        # Ensure prefix ends with "/" if provided
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        self._prefix = prefix
        
    @property
    def run(self) -> Any:
        """Access underlying wandb.Run if needed."""
        return self._wandb_run
    
    def log_metrics(
        self, 
        metrics: Dict[str, float], 
        step: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """
        Log metrics to WandB with optional prefix and step.
        
        Args:
            metrics: Dictionary of metric name → value
            step: Optional step number for time-series
            prefix: Additional prefix (combined with instance prefix)
        """
        # Combine instance prefix with call-time prefix
        full_prefix = self._prefix + prefix
        
        if full_prefix:
            metrics = {f"{full_prefix}{k}": v for k, v in metrics.items()}
        
        self._wandb_run.log(metrics, step=step)
        
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters to WandB config.
        
        Args:
            params: Dictionary of parameter name → value
        """
        if self._prefix:
            params = {f"{self._prefix}{k}": v for k, v in params.items()}
        
        self._wandb_run.config.update(params, allow_val_change=True)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"run={self._wandb_run.id}, "
            f"prefix={self._prefix!r})"
        )


class NoOpLogger:
    """
    No-op logger that does nothing.
    
    Used when logging is disabled or not provided. Satisfies LoggerProtocol
    without performing any actual logging operations.
    """
    
    def log_metrics(
        self, 
        metrics: Dict[str, float], 
        step: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """No-op metric logging."""
        pass
    
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """No-op hyperparameter logging."""
        pass
    
    def __repr__(self) -> str:
        return "NoOpLogger()"


def create_logger(
    wandb_run: Optional[Any] = None,
    prefix: str = "",
    logger_type: str = "auto"
) -> Union[DirectWandbLogger, NoOpLogger]:
    """
    Factory for creating appropriate logger based on context.
    
    Args:
        wandb_run: Optional existing wandb.Run to wrap
        prefix: Optional prefix for all logged keys
        logger_type: Type of logger ("auto", "direct", "noop")
        
    Returns:
        Appropriate logger instance
        
    Example:
        >>> # Auto-detect: creates DirectWandbLogger if run provided
        >>> logger = create_logger(wandb_run, prefix="step_0/")
        >>> 
        >>> # Explicit no-op
        >>> logger = create_logger(logger_type="noop")
    """
    if logger_type == "noop":
        return NoOpLogger()
    
    if wandb_run is not None:
        return DirectWandbLogger(wandb_run, prefix=prefix)
    
    # No run provided, default to no-op
    logger.info("No WandB run provided, using NoOpLogger")
    return NoOpLogger()
