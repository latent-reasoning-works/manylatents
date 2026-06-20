"""Tests for MIOFlow LightningModule."""

import functools

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
            accelerator="cpu",  # MIOFlow needs CPU: torchdiffeq float64 + cdist_backward unsupported on MPS
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
        )
        trainer.fit(mioflow_module, mioflow_module.datamodule.train_dataloader())

        # After training, encode should work
        z = mioflow_module.encode(time_labeled_batch["data"])
        assert z.shape == (60, 5)

    def test_training_reduces_distribution_distance(self, mioflow_module, time_labeled_batch):
        """MIOFlow must actually LEARN, not just run.

        On an easy time-resolved problem (mean-shift per timestep, tiny variance)
        the flow that maps each population to the next exists, so the per-interval
        W2^2 between the predicted and true next population must drop substantially
        after training. Shape/finite/grad checks (the other tests) would pass for a
        non-learning impl; this one would not. This guards the correctness that the
        original "not battle-tested" removal left unverified.
        """
        import ot
        from torchdiffeq import odeint

        mioflow_module.setup()
        groups = mioflow_module._group_by_time(time_labeled_batch)

        def per_interval_w2sq(net):
            errs = []
            for i in range(len(groups) - 1):
                xs, ts = groups[i]
                xe, te = groups[i + 1]
                with torch.no_grad():
                    pred = odeint(net, xs, torch.tensor([ts, te], dtype=torch.float32))[1]
                M = torch.cdist(pred, xe) ** 2
                errs.append(
                    float(ot.emd2(torch.as_tensor(ot.unif(pred.size(0))),
                                  torch.as_tensor(ot.unif(xe.size(0))), M))
                )
            return errs

        before = per_interval_w2sq(mioflow_module.network)

        # Train the global objective directly (deterministic, fast).
        opt = torch.optim.Adam(mioflow_module.network.parameters(), lr=1e-3)
        for _ in range(150):
            opt.zero_grad()
            loss = mioflow_module._global_step(groups)["loss"]
            loss.backward()
            opt.step()

        after = per_interval_w2sq(mioflow_module.network)

        # Every interval should improve, and the total distance should drop hard.
        assert sum(after) < 0.6 * sum(before), (
            f"training failed to reduce distribution distance: {before} -> {after}"
        )
        for b, a in zip(before, after):
            assert a < b, f"interval distance did not improve: {b:.3f} -> {a:.3f}"


class TestMIOFlowThroughRunExperiment:
    """End-to-end: mioflow as a runnable op via run_experiment (issue #274).

    Mirrors TestLatentODEThroughRunExperiment (#269). The new surface MIOFlow
    exercises that latent_ode doesn't: the batch must carry a per-sample TIME
    label, so this uses a small time-resolved datamodule.
    """

    def test_run_experiment_returns_latent_outputs(self):
        import numpy as np
        from lightning import LightningDataModule, Trainer
        from torch.utils.data import DataLoader, Dataset

        from manylatents.algorithms.lightning.mioflow import MIOFlow
        from manylatents.algorithms.lightning.networks.mioflow_net import MIOFlowODEFunc
        from manylatents.callbacks.embedding.base import validate_latent_outputs
        from manylatents.experiment import run_experiment

        # Toy time-resolved input: 3 timepoints, mean-shift per step, 5 dims.
        torch.manual_seed(0)
        parts_x, parts_y = [], []
        for t in (0.0, 0.5, 1.0):
            parts_x.append(torch.randn(20, 5) * 0.3 + t)
            parts_y.append(torch.full((20,), t))
        X = torch.cat(parts_x)
        Y = torch.cat(parts_y)

        class TimeLabeledDataset(Dataset):
            def __len__(self):
                return len(X)

            def __getitem__(self, idx):
                # "label" is the manyLatents op-contract key; for MIOFlow it is
                # the per-sample timepoint.
                return {"data": X[idx], "label": Y[idx]}

            def get_labels(self):
                return Y.numpy()

        class TimeLabeledDataModule(LightningDataModule):
            def setup(self, stage=None):
                self.train_dataset = TimeLabeledDataset()
                self.test_dataset = TimeLabeledDataset()

            def _loader(self):
                return DataLoader(TimeLabeledDataset(), batch_size=len(X))

            def train_dataloader(self):
                return self._loader()

            def val_dataloader(self):
                return self._loader()

            def test_dataloader(self):
                return self._loader()

        datamodule = TimeLabeledDataModule()

        net = MIOFlowODEFunc(input_dim=5, hidden_dim=16)
        model = MIOFlow(
            network=net,
            optimizer=functools.partial(torch.optim.Adam, lr=1e-3),
            n_local_epochs=0,
            n_global_epochs=2,
            n_post_local_epochs=0,
            lambda_ot=1.0,
            lambda_energy=0.01,
            init_seed=0,
        )
        model.datamodule = datamodule  # lets encode() infer the time span

        # accelerator="cpu" is deliberate: torchdiffeq's dopri5 builds float64
        # solver tolerances and the OT/density losses use cdist, whose backward
        # is unimplemented on MPS. The default "auto" would pick MPS on macOS and
        # crash. (Captured as M1 friction — same as latent_ode.)
        trainer = Trainer(
            accelerator="cpu", devices=1, max_epochs=2, logger=False,
            enable_checkpointing=False, enable_progress_bar=False,
        )

        result = run_experiment(
            datamodule=datamodule, algorithm=model, trainer=trainer, seed=0
        )

        validate_latent_outputs(result)
        emb = result["embeddings"]
        assert emb.ndim == 2, f"expected 2-D embeddings, got shape {emb.shape}"
        assert emb.shape == (60, 5), f"unexpected embedding shape {emb.shape}"
        assert np.isfinite(emb).all(), "embeddings contain NaN/Inf"
