"""Lightning callbacks for activation tracking and diffusion operators."""
from manylatents.lightning.callbacks.activation_tracker import (
    ProbeTrigger,
    ActivationTrajectoryCallback,
)
from manylatents.lightning.callbacks.wandb_probe import WandbProbeLogger

# Backward-compatible alias
RepresentationProbeCallback = ActivationTrajectoryCallback

__all__ = ["ProbeTrigger", "ActivationTrajectoryCallback", "RepresentationProbeCallback", "WandbProbeLogger"]
