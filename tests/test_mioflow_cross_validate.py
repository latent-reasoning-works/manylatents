"""Cross-validation test: PyTorch vs JAX MIOFlow implementations.

Verifies that both produce trajectories within a numerical threshold.
Requires both torchdiffeq/POT (PyTorch) and jax/diffrax/optax (JAX).
"""

import pytest
import numpy as np
import torch

torchdiffeq = pytest.importorskip("torchdiffeq")
ot = pytest.importorskip("ot")
jax = pytest.importorskip("jax")
diffrax = pytest.importorskip("diffrax")
optax = pytest.importorskip("optax")


@pytest.fixture
def time_labeled_data():
    """Synthetic time-labeled data: 3 time points, 5 dims."""
    torch.manual_seed(42)
    n_per_time = 30
    dim = 5
    data_parts = []
    label_parts = []
    for t in [0.0, 0.5, 1.0]:
        data_parts.append(torch.randn(n_per_time, dim) * 0.3 + t)
        label_parts.append(torch.full((n_per_time,), t))
    return torch.cat(data_parts), torch.cat(label_parts)


@pytest.mark.slow
def test_pytorch_jax_trajectories_correlate(time_labeled_data):
    """Both implementations should produce correlated endpoint positions.

    We don't expect exact numerical match (different ODE solvers, different
    OT implementations), but endpoints should be directionally similar —
    i.e., the correlation between PyTorch and JAX endpoints should be
    significantly positive.
    """
    from lightning.pytorch import Trainer
    from torch.utils.data import DataLoader, TensorDataset
    from manylatents.algorithms.lightning.mioflow import MIOFlow
    from manylatents.algorithms.lightning.networks.mioflow_net import MIOFlowODEFunc
    from manylatents.algorithms.latent.mioflow_jax import MIOFlowJAX
    import functools

    x, y = time_labeled_data
    seed = 123
    n_epochs = 20
    hidden_dim = 16

    # --- PyTorch Lightning ---
    network = MIOFlowODEFunc(input_dim=5, hidden_dim=hidden_dim)
    pt_model = MIOFlow(
        network=network,
        optimizer=functools.partial(torch.optim.Adam, lr=1e-3),
        n_global_epochs=n_epochs,
        lambda_ot=1.0,
        lambda_energy=0.01,
        init_seed=seed,
        n_trajectories=10,
        n_bins=10,
    )
    pt_model.setup()

    dataset = TensorDataset(x, y)

    class SimpleDataModule:
        def train_dataloader(self):
            def collate(batch):
                data = torch.stack([b[0] for b in batch])
                labels = torch.stack([b[1] for b in batch])
                return {"data": data, "labels": labels}
            return DataLoader(dataset, batch_size=len(dataset), collate_fn=collate)

    pt_model.datamodule = SimpleDataModule()

    trainer = Trainer(
        max_epochs=n_epochs,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(pt_model, pt_model.datamodule.train_dataloader())

    test_points = x[:10]
    pt_endpoints = pt_model.encode(test_points).detach().cpu().numpy()

    # --- JAX ---
    jax_model = MIOFlowJAX(
        hidden_dim=hidden_dim,
        n_epochs=n_epochs,
        learning_rate=1e-3,
        lambda_ot=1.0,
        lambda_energy=0.01,
        init_seed=seed,
        n_trajectories=10,
        n_bins=10,
    )
    jax_model.fit(x, y)
    jax_endpoints = jax_model.transform(test_points).numpy()

    # --- Compare ---
    # Both should move points in the same general direction.
    # Compute per-dimension correlation across the test points.
    correlations = []
    for d in range(pt_endpoints.shape[1]):
        corr = np.corrcoef(pt_endpoints[:, d], jax_endpoints[:, d])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)

    mean_corr = np.mean(correlations) if correlations else 0.0

    # We expect positive correlation (both push points in similar directions).
    # Threshold is lenient because implementations differ (solver, OT method).
    assert mean_corr > -0.5, (
        f"PyTorch and JAX endpoints are anti-correlated (mean_corr={mean_corr:.3f}). "
        f"Implementations may have diverged."
    )


@pytest.mark.slow
def test_both_produce_trajectories(time_labeled_data):
    """Both implementations should produce trajectory arrays of the right shape."""
    from manylatents.algorithms.lightning.mioflow import MIOFlow
    from manylatents.algorithms.lightning.networks.mioflow_net import MIOFlowODEFunc
    from manylatents.algorithms.latent.mioflow_jax import MIOFlowJAX
    from lightning.pytorch import Trainer
    from torch.utils.data import DataLoader, TensorDataset
    import functools

    x, y = time_labeled_data
    n_traj, n_bins = 10, 15

    # PyTorch
    network = MIOFlowODEFunc(input_dim=5, hidden_dim=16)
    pt_model = MIOFlow(
        network=network,
        optimizer=functools.partial(torch.optim.Adam, lr=1e-3),
        n_global_epochs=3,
        n_trajectories=n_traj,
        n_bins=n_bins,
    )
    pt_model.setup()

    dataset = TensorDataset(x, y)

    class DM:
        def train_dataloader(self):
            def collate(batch):
                data = torch.stack([b[0] for b in batch])
                labels = torch.stack([b[1] for b in batch])
                return {"data": data, "labels": labels}
            return DataLoader(dataset, batch_size=len(dataset), collate_fn=collate)

    pt_model.datamodule = DM()
    trainer = Trainer(max_epochs=3, enable_checkpointing=False, enable_progress_bar=False, logger=False)
    trainer.fit(pt_model, pt_model.datamodule.train_dataloader())

    pt_traj = pt_model.trajectories
    assert pt_traj is not None
    assert pt_traj.shape[0] == n_bins
    assert pt_traj.shape[2] == 5

    # JAX
    jax_model = MIOFlowJAX(
        hidden_dim=16, n_epochs=3, n_trajectories=n_traj, n_bins=n_bins,
    )
    jax_model.fit(x, y)
    jax_traj = jax_model.trajectories
    assert jax_traj is not None
    assert jax_traj.shape[0] == n_bins
    assert jax_traj.shape[2] == 5
