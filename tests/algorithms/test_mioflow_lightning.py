"""Tests for MIOFlow LightningModule."""

import pytest
import torch
from torch import Tensor

pytest.importorskip("torchdiffeq")
ot = pytest.importorskip("ot")


class TestMIOFlowLightningModule:
    """Tests for the MIOFlow LightningModule training wrapper."""

    @pytest.fixture
    def time_labeled_batch(self):
        """Create a batch with time-labeled data (3 time points)."""
        torch.manual_seed(42)
        n_per_time = 20
        dim = 5
        data_parts = []
        label_parts = []
        for t in [0.0, 0.5, 1.0]:
            data_parts.append(torch.randn(n_per_time, dim) + t)
            label_parts.append(torch.full((n_per_time,), t))
        return {
            "data": torch.cat(data_parts),
            "labels": torch.cat(label_parts),
        }

    @pytest.fixture
    def mioflow_module(self):
        """Create a MIOFlow module with direct network config."""
        from manylatents.algorithms.lightning.mioflow import MIOFlow
        from manylatents.algorithms.lightning.networks.mioflow_net import MIOFlowODEFunc
        import functools

        network = MIOFlowODEFunc(input_dim=5, hidden_dim=16)
        optimizer = functools.partial(torch.optim.Adam, lr=1e-3)

        return MIOFlow(
            network=network,
            optimizer=optimizer,
            n_local_epochs=2,
            n_global_epochs=3,
            n_post_local_epochs=1,
            lambda_ot=1.0,
            lambda_energy=0.01,
            lambda_density=0.0,
            energy_time_steps=5,
            sample_size=10,
            n_trajectories=10,
            n_bins=10,
        )

    def test_setup_with_direct_network(self, mioflow_module):
        """Test that setup works with directly passed network."""
        mioflow_module.setup()
        assert mioflow_module.network is not None

    def test_training_mode_selection(self, mioflow_module):
        """Test that training mode switches based on epoch."""
        # n_local=2, n_global=3, n_post_local=1
        assert mioflow_module._get_training_mode(0) == "local"
        assert mioflow_module._get_training_mode(1) == "local"
        assert mioflow_module._get_training_mode(2) == "global"
        assert mioflow_module._get_training_mode(4) == "global"
        assert mioflow_module._get_training_mode(5) == "local"  # post-local

    def test_group_by_time(self, mioflow_module, time_labeled_batch):
        """Test that batch data is correctly grouped by time labels."""
        mioflow_module.setup()
        groups = mioflow_module._group_by_time(time_labeled_batch)
        assert len(groups) == 3
        assert groups[0][1] == 0.0
        assert groups[1][1] == 0.5
        assert groups[2][1] == 1.0
        assert groups[0][0].shape == (20, 5)

    def test_local_step(self, mioflow_module, time_labeled_batch):
        """Test local training step returns valid loss dict."""
        mioflow_module.setup()
        groups = mioflow_module._group_by_time(time_labeled_batch)
        result = mioflow_module._local_step(groups)
        assert "loss" in result
        assert result["loss"].requires_grad
        assert torch.isfinite(result["loss"])

    def test_global_step(self, mioflow_module, time_labeled_batch):
        """Test global training step returns valid loss dict."""
        mioflow_module.setup()
        groups = mioflow_module._group_by_time(time_labeled_batch)
        result = mioflow_module._global_step(groups)
        assert "loss" in result
        assert result["loss"].requires_grad
        assert torch.isfinite(result["loss"])

    def test_encode_returns_correct_shape(self, mioflow_module):
        """Test that encode() returns (n, d) embeddings."""
        mioflow_module.setup()
        x = torch.randn(15, 5)
        z = mioflow_module.encode(x)
        assert z.shape == (15, 5)
        assert not z.requires_grad

    def test_total_epochs(self, mioflow_module):
        """Test total epoch calculation."""
        assert mioflow_module.total_epochs == 6  # 2 + 3 + 1

    def test_end_to_end_training(self, mioflow_module, time_labeled_batch):
        """Test end-to-end training with Lightning Trainer."""
        from lightning.pytorch import Trainer
        from torch.utils.data import DataLoader, TensorDataset

        mioflow_module.setup()

        # Create a simple dataloader
        dataset = TensorDataset(
            time_labeled_batch["data"],
            time_labeled_batch["labels"],
        )

        class SimpleDataModule:
            def train_dataloader(self):
                def collate(batch):
                    data = torch.stack([b[0] for b in batch])
                    labels = torch.stack([b[1] for b in batch])
                    return {"data": data, "labels": labels}

                return DataLoader(dataset, batch_size=len(dataset), collate_fn=collate)

        mioflow_module.datamodule = SimpleDataModule()

        trainer = Trainer(
            max_epochs=3,
            fast_dev_run=True,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
        )
        trainer.fit(mioflow_module, mioflow_module.datamodule.train_dataloader())

        # After training, encode should work
        z = mioflow_module.encode(time_labeled_batch["data"])
        assert z.shape == (60, 5)
