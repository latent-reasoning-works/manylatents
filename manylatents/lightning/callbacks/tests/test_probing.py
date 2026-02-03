# manylatents/lightning/callbacks/tests/test_probing.py
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lightning import LightningModule, Trainer
from manylatents.lightning.callbacks.probing import (
    ProbeTrigger,
    RepresentationProbeCallback,
)
from manylatents.lightning.hooks import LayerSpec


def test_probe_trigger_step_based():
    trigger = ProbeTrigger(every_n_steps=100)

    assert trigger.should_fire(step=0, epoch=0) is True   # First step
    assert trigger.should_fire(step=50, epoch=0) is False
    assert trigger.should_fire(step=100, epoch=0) is True
    assert trigger.should_fire(step=200, epoch=0) is True


def test_probe_trigger_epoch_based():
    trigger = ProbeTrigger(every_n_epochs=2)

    assert trigger.should_fire(step=0, epoch=0, epoch_end=True) is True
    assert trigger.should_fire(step=0, epoch=1, epoch_end=True) is False
    assert trigger.should_fire(step=0, epoch=2, epoch_end=True) is True


def test_probe_trigger_combined():
    trigger = ProbeTrigger(every_n_steps=100, every_n_epochs=1)

    # Steps trigger
    assert trigger.should_fire(step=100, epoch=0) is True
    # Epoch also triggers
    assert trigger.should_fire(step=50, epoch=1, epoch_end=True) is True


def test_probe_trigger_disabled():
    trigger = ProbeTrigger()  # No triggers set

    assert trigger.should_fire(step=100, epoch=5) is False


# RepresentationProbeCallback tests


class TinyModel(LightningModule):
    """Minimal model for testing."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = nn.functional.mse_loss(out, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def make_probe_loader(n_samples=20, input_dim=10):
    x = torch.randn(n_samples, input_dim)
    y = torch.randn(n_samples, 5)
    return DataLoader(TensorDataset(x, y), batch_size=10)


def test_representation_probe_callback_captures():
    """Callback should capture activations and compute diffusion ops."""
    model = TinyModel()
    probe_loader = make_probe_loader()

    callback = RepresentationProbeCallback(
        probe_loader=probe_loader,
        layer_specs=[LayerSpec(path="0", reduce="none")],  # Path relative to .network
        trigger=ProbeTrigger(every_n_steps=1),
    )

    train_loader = make_probe_loader(n_samples=40)

    trainer = Trainer(
        max_epochs=1,
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, train_loader)

    # Should have captured at least one trajectory point
    trajectory = callback.get_trajectory()
    assert len(trajectory) > 0

    # Each point should have step and diffusion operator
    step, diff_op = trajectory[0]
    assert isinstance(step, int)
    assert isinstance(diff_op, np.ndarray)
    assert diff_op.shape == (20, 20)  # probe_loader has 20 samples


def test_representation_probe_callback_multi_layer():
    """Should capture multiple layers."""
    model = TinyModel()
    probe_loader = make_probe_loader()

    callback = RepresentationProbeCallback(
        probe_loader=probe_loader,
        layer_specs=[
            LayerSpec(path="0", reduce="none"),  # First linear
            LayerSpec(path="2", reduce="none"),  # Second linear
        ],
        trigger=ProbeTrigger(every_n_steps=2),
    )

    train_loader = make_probe_loader(n_samples=40)

    trainer = Trainer(
        max_epochs=1,
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, train_loader)

    trajectory = callback.get_trajectory()
    assert len(trajectory) > 0

    # Multi-layer returns dict
    step, ops = trajectory[0]
    assert isinstance(ops, dict)
    assert "0" in ops
    assert "2" in ops


def test_representation_probe_callback_epoch_trigger():
    """Should trigger at epoch end."""
    model = TinyModel()
    probe_loader = make_probe_loader()

    callback = RepresentationProbeCallback(
        probe_loader=probe_loader,
        layer_specs=[LayerSpec(path="0", reduce="none")],
        trigger=ProbeTrigger(every_n_epochs=1),
    )

    train_loader = make_probe_loader(n_samples=40)

    trainer = Trainer(
        max_epochs=2,
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, train_loader)

    trajectory = callback.get_full_trajectory()
    # Should have captured at epoch 0 and epoch 1
    epochs = [p.epoch for p in trajectory]
    assert 0 in epochs
    assert 1 in epochs
