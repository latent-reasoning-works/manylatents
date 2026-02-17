"""
Adapters for PyTorch Lightning loggers.

These adapters allow reusing external logging contexts (e.g., parent WandB runs)
within PyTorch Lightning trainers, enabling hierarchical logging without creating
duplicate runs or conflicting contexts.

Usage:
    # Parent creates logging context
    wandb_run = wandb.init(project="my_project", name="workflow")
    
    # Child reuses parent's context with optional prefix
    lightning_logger = WandbRunAdapter(wandb_run, prefix="step_0/")
    trainer = Trainer(logger=lightning_logger)
    
    # All Lightning logs go to parent's run with prefix
    trainer.fit(model, datamodule)
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING, Union

try:
    import wandb
    wandb.init  # verify real package, not wandb/ output dir
except (ImportError, AttributeError):
    wandb = None

if TYPE_CHECKING:
    import wandb as wandb_types

from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class WandbRunAdapter(Logger):
    """
    PyTorch Lightning logger that adapts an existing wandb.Run.
    
    This adapter allows multiple PyTorch Lightning trainers to log to the same
    WandB run, enabling hierarchical logging patterns where a parent orchestrator
    creates a single run and child components log to it with optional prefixes.
    
    Unlike the standard WandbLogger which creates its own wandb.init() call,
    this adapter wraps an existing wandb.Run, preventing duplicate runs and
    allowing for better organization in experiments with multiple training steps.
    
    Args:
        wandb_run: An existing wandb.Run instance to log to
        prefix: Optional prefix for all logged keys (e.g., "step_0/" or "train/")
        
    Example:
        >>> # Parent context
        >>> parent_run = wandb.init(project="geomancer", name="workflow")
        >>> 
        >>> # Step 1: PCA training
        >>> logger1 = WandbRunAdapter(parent_run, prefix="pca/")
        >>> trainer1 = Trainer(logger=logger1)
        >>> trainer1.fit(pca_model, datamodule)
        >>> 
        >>> # Step 2: UMAP training
        >>> logger2 = WandbRunAdapter(parent_run, prefix="umap/")
        >>> trainer2 = Trainer(logger=logger2)
        >>> trainer2.fit(umap_model, datamodule)
        >>> 
        >>> # Both log to same run with different prefixes
        >>> parent_run.finish()
    """
    
    def __init__(
        self, 
        wandb_run: wandb.Run, 
        prefix: str = ""
    ):
        """
        Initialize the adapter with an existing WandB run.
        
        Args:
            wandb_run: Existing wandb.Run to wrap
            prefix: Optional prefix for all log keys. If provided and doesn't
                   end with "/", one will be added automatically.
        """
        super().__init__()
        self._experiment = wandb_run
        
        # Ensure prefix ends with "/" if provided
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        self._prefix = prefix
        
    @property
    def name(self) -> str:
        """Return the name of the wrapped run."""
        return self._experiment.name
    
    @property
    def version(self) -> str:
        """Return the ID of the wrapped run."""
        return self._experiment.id
    
    @property
    @rank_zero_experiment
    def experiment(self) -> wandb.Run:
        """
        Return the wrapped wandb.Run.
        
        This property satisfies Lightning's logger interface and provides
        access to the underlying WandB run for advanced use cases.
        """
        return self._experiment
    
    @rank_zero_only
    def log_metrics(
        self, 
        metrics: Mapping[str, float], 
        step: Optional[int] = None
    ) -> None:
        """
        Log metrics to the wrapped WandB run with optional prefix.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number for time-series logging
        """
        if self._prefix:
            metrics = {f"{self._prefix}{k}": v for k, v in metrics.items()}
        
        self._experiment.log(metrics, step=step)
    
    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Mapping]) -> None:
        """
        Log hyperparameters to the wrapped WandB run's config.
        
        Args:
            params: Dictionary of hyperparameter names to values
        """
        # Convert to dict if needed
        if not isinstance(params, dict):
            params = dict(params)
        
        # Add prefix to param names if specified
        if self._prefix:
            params = {f"{self._prefix}{k}": v for k, v in params.items()}
        
        # Update config (merges with existing)
        self._experiment.config.update(params, allow_val_change=True)
    
    @rank_zero_only
    def finalize(self, status: str) -> None:
        """
        Finalize the logger.
        
        Note: This does NOT call wandb.finish() on the wrapped run, since
        the parent context owns the run lifecycle. The parent is responsible
        for calling finish() when all child loggers are done.
        
        Args:
            status: Final status string (e.g., "success", "failed")
        """
        # Intentionally do nothing - parent manages run lifecycle
        pass
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"version={self.version!r}, "
            f"prefix={self._prefix!r})"
        )


# Future: Add other adapters as needed
# class MLflowRunAdapter(Logger):
#     """Adapter for MLflow runs."""
#     pass
#
# class NeptuneRunAdapter(Logger):
#     """Adapter for Neptune runs."""
#     pass
