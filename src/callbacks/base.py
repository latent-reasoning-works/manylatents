from abc import ABC

from omegaconf import DictConfig


class BaseCallback(ABC):
    def on_experiment_start(self, cfg: DictConfig) -> None:
        """Called at the beginning of the experiment."""
        pass

    def on_experiment_end(self) -> None:
        """Called at the end of the experiment."""
        pass

    def on_dr_end(self, original_data, embeddings) -> None:
        """Called when DR processing completes."""
        pass

    def on_training_start(self) -> None:
        """Called when training starts."""
        pass

    def on_training_end(self) -> None:
        """Called when training ends."""
        pass