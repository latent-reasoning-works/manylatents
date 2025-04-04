## boilerplate code for now:
## intended to be used to combine multiple callbacks into one
## dr or lightning module callbacks

from typing import List

from src.callbacks.cb_base import BaseCallback


class CompositeCallback(BaseCallback):
    def __init__(self, callbacks: List[BaseCallback]):
        self.callbacks = callbacks

    def on_experiment_start(self, cfg):
        for cb in self.callbacks:
            if hasattr(cb, "on_experiment_start"):
                cb.on_experiment_start(cfg)

    def on_dr_end(self, original_data, embeddings):
        for cb in self.callbacks:
            if hasattr(cb, "on_dr_end"):
                cb.on_dr_end(original_data, embeddings)

    def on_training_start(self):
        for cb in self.callbacks:
            if hasattr(cb, "on_training_start"):
                cb.on_training_start()

    def on_training_end(self):
        for cb in self.callbacks:
            if hasattr(cb, "on_training_end"):
                cb.on_training_end()

    def on_experiment_end(self):
        for cb in self.callbacks:
            if hasattr(cb, "on_experiment_end"):
                cb.on_experiment_end()
