"""Representation probing callback for Lightning."""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from torch import Tensor
from torch.utils.data import DataLoader

from manylatents.lightning.hooks import ActivationExtractor, LayerSpec
from manylatents.callbacks.probing import DiffusionGauge
from manylatents.lightning.callbacks.wandb_probe import WandbProbeLogger

logger = logging.getLogger(__name__)


@dataclass
class ProbeTrigger:
    """Configuration for when to trigger representation probes.

    Multiple triggers can be combined (OR logic).

    Attributes:
        every_n_steps: Trigger every N training steps
        every_n_epochs: Trigger at the end of every N epochs
        on_checkpoint: Trigger when checkpoint is saved
        on_validation_end: Trigger after validation
    """
    every_n_steps: Optional[int] = None
    every_n_epochs: Optional[int] = None
    on_checkpoint: bool = False
    on_validation_end: bool = False

    def should_fire(
        self,
        step: int,
        epoch: int,
        epoch_end: bool = False,
        checkpoint: bool = False,
        validation_end: bool = False,
    ) -> bool:
        """Check if audit should trigger based on current state."""
        if self.every_n_steps is not None:
            if step % self.every_n_steps == 0:
                return True

        if self.every_n_epochs is not None and epoch_end:
            if epoch % self.every_n_epochs == 0:
                return True

        if self.on_checkpoint and checkpoint:
            return True

        if self.on_validation_end and validation_end:
            return True

        return False


@dataclass
class TrajectoryPoint:
    """A single point in the representation trajectory."""
    step: int
    epoch: int
    diffusion_operators: Dict[str, np.ndarray]  # layer_path -> operator
    metadata: Dict[str, Any] = field(default_factory=dict)


class RepresentationProbeCallback(Callback):
    """Callback that extracts activations and computes diffusion operators.

    At configurable triggers during training, this callback:
    1. Runs the probe set through the model
    2. Extracts activations from specified layers
    3. Computes diffusion operators from activations
    4. Stores (step, operator) pairs in a trajectory

    Usage:
        # Option 1: Pass probe_loader directly (for testing)
        callback = RepresentationProbeCallback(
            probe_loader=probe_loader,
            layer_specs=[LayerSpec("transformer.h[-1]")],
            trigger=ProbeTrigger(every_n_steps=100),
        )

        # Option 2: Let callback fetch from datamodule (for experiments)
        callback = RepresentationProbeCallback(
            layer_specs=[LayerSpec("transformer.h[-1]")],
            trigger=ProbeTrigger(every_n_steps=100),
        )
        # Datamodule must have probe_dataloader() method

        trainer = Trainer(callbacks=[callback])
        trainer.fit(model)

        trajectory = callback.get_trajectory()
    """

    def __init__(
        self,
        layer_specs: List[LayerSpec],
        trigger: ProbeTrigger,
        probe_loader: Optional[DataLoader] = None,
        gauge: Optional[DiffusionGauge] = None,
        log_to_wandb: bool = False,
        wandb_logger: Optional[WandbProbeLogger] = None,
    ):
        super().__init__()
        self.probe_loader = probe_loader
        self.layer_specs = layer_specs
        self.trigger = trigger
        self.gauge = gauge or DiffusionGauge()
        self.log_to_wandb = log_to_wandb
        self.wandb_logger = wandb_logger or WandbProbeLogger() if log_to_wandb else None

        self.extractor = ActivationExtractor(layer_specs)
        self._trajectory: List[TrajectoryPoint] = []

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        """Fetch probe_loader from datamodule if not provided."""
        logger.info("RepresentationProbeCallback.on_fit_start called")
        if self.probe_loader is None:
            dm = trainer.datamodule
            if dm is not None and hasattr(dm, 'probe_dataloader'):
                self.probe_loader = dm.probe_dataloader()
                logger.info(f"Probe loader fetched from datamodule: {len(self.probe_loader)} batches")
            else:
                raise ValueError(
                    "probe_loader not provided and datamodule has no probe_dataloader() method. "
                    "Either pass probe_loader to RepresentationProbeCallback or use a datamodule "
                    "with probe_dataloader() (e.g., TextDataModule)."
                )

    def _extract_and_gauge(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> Dict[str, np.ndarray]:
        """Extract activations and compute diffusion operators."""
        # Get the underlying network
        network = getattr(pl_module, 'network', pl_module)

        # Run probe set through model
        network.eval()
        with torch.no_grad():
            with self.extractor.capture(network):
                for batch in self.probe_loader:
                    # Handle different batch formats
                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0]
                    elif isinstance(batch, dict):
                        inputs = batch
                    else:
                        inputs = batch

                    if isinstance(inputs, torch.Tensor):
                        inputs = inputs.to(pl_module.device)
                        network(inputs)
                    else:
                        # Dict input (for HF models)
                        inputs = {k: v.to(pl_module.device) for k, v in inputs.items()}
                        network(**inputs)

        network.train()

        # Get activations and compute diffusion operators
        activations = self.extractor.get_activations()
        diffusion_ops = {}
        for path, acts in activations.items():
            diffusion_ops[path] = self.gauge(acts)

        return diffusion_ops

    def _maybe_probe(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        epoch_end: bool = False,
        checkpoint: bool = False,
        validation_end: bool = False,
    ):
        """Check trigger and perform audit if needed."""
        should_fire = self.trigger.should_fire(
            step=trainer.global_step,
            epoch=trainer.current_epoch,
            epoch_end=epoch_end,
            checkpoint=checkpoint,
            validation_end=validation_end,
        )
        logger.debug(f"_maybe_probe: step={trainer.global_step}, should_fire={should_fire}")
        if not should_fire:
            return

        logger.info(f"Probe triggered at step {trainer.global_step}")
        diff_ops = self._extract_and_gauge(trainer, pl_module)

        point = TrajectoryPoint(
            step=trainer.global_step,
            epoch=trainer.current_epoch,
            diffusion_operators=diff_ops,
        )
        self._trajectory.append(point)

        # Log to wandb if enabled
        if self.log_to_wandb and self.wandb_logger is not None:
            self.wandb_logger.log_trajectory(
                self.get_trajectory(),
                step=trainer.global_step,
            )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        """Check step-based triggers."""
        self._maybe_probe(trainer, pl_module)

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ):
        """Check epoch-based triggers."""
        self._maybe_probe(trainer, pl_module, epoch_end=True)

    def on_validation_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ):
        """Check validation-end trigger."""
        self._maybe_probe(trainer, pl_module, validation_end=True)

    def get_trajectory(self) -> List[tuple]:
        """Get trajectory as list of (step, diffusion_operator) tuples.

        For single-layer extraction, returns the operator directly.
        For multi-layer, returns dict of operators.
        """
        result = []
        for point in self._trajectory:
            if len(point.diffusion_operators) == 1:
                # Single layer - return operator directly
                op = list(point.diffusion_operators.values())[0]
                result.append((point.step, op))
            else:
                result.append((point.step, point.diffusion_operators))
        return result

    def get_full_trajectory(self) -> List[TrajectoryPoint]:
        """Get full trajectory with metadata."""
        return self._trajectory.copy()

    def clear(self):
        """Clear stored trajectory."""
        self._trajectory.clear()
