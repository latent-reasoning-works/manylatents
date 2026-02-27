"""Tests for GAGA network components and loss functions."""

import pytest
import torch


# ---------------------------------------------------------------------------
# Network tests
# ---------------------------------------------------------------------------


class TestPreprocessor:
    """Tests for the Preprocessor normalization module."""

    def test_normalize_unnormalize_roundtrip(self):
        """normalize then unnormalize recovers original data."""
        from manylatents.algorithms.lightning.networks.gaga_net import Preprocessor

        pp = Preprocessor(mean=2.0, std=3.0, dist_std=5.0)
        x = torch.randn(16, 10)
        reconstructed = pp.unnormalize(pp.normalize(x))
        assert torch.allclose(reconstructed, x, atol=1e-5)

    def test_normalize_zero_mean_unit_var(self):
        """Normalized output has the expected shift and scale."""
        from manylatents.algorithms.lightning.networks.gaga_net import Preprocessor

        mean = torch.tensor([1.0, 2.0])
        std = torch.tensor([0.5, 0.5])
        pp = Preprocessor(mean=mean, std=std)
        x = torch.tensor([[1.0, 2.0], [1.5, 2.5]])
        normed = pp.normalize(x)
        assert torch.allclose(normed[0], torch.zeros(2), atol=1e-6)

    def test_normalize_dist(self):
        """Distance normalization divides by dist_std."""
        from manylatents.algorithms.lightning.networks.gaga_net import Preprocessor

        pp = Preprocessor(dist_std=4.0)
        d = torch.tensor([8.0, 12.0])
        assert torch.allclose(pp.normalize_dist(d), torch.tensor([2.0, 3.0]))

    def test_buffers_move_with_device(self):
        """Buffers follow .to() calls (smoke test on CPU)."""
        from manylatents.algorithms.lightning.networks.gaga_net import Preprocessor

        pp = Preprocessor(mean=1.0, std=2.0, dist_std=3.0)
        pp = pp.to("cpu")
        assert pp.mean.device.type == "cpu"

    def test_state_dict_contains_buffers(self):
        """All three buffers appear in state_dict."""
        from manylatents.algorithms.lightning.networks.gaga_net import Preprocessor

        pp = Preprocessor(mean=1.0, std=2.0, dist_std=3.0)
        sd = pp.state_dict()
        assert "mean" in sd and "std" in sd and "dist_std" in sd


class TestGAGANetwork:
    """Tests for the GAGANetwork."""

    def test_forward_shape(self):
        """forward returns (x_hat, z) with correct shapes."""
        from manylatents.algorithms.lightning.networks.gaga_net import GAGANetwork

        net = GAGANetwork(input_dim=50, latent_dim=2, hidden_dims=[32, 16])
        x = torch.randn(20, 50)
        x_hat, z = net(x)
        assert x_hat.shape == (20, 50)
        assert z.shape == (20, 2)

    def test_encode_shape(self):
        """encode returns correct latent shape."""
        from manylatents.algorithms.lightning.networks.gaga_net import GAGANetwork

        net = GAGANetwork(input_dim=30, latent_dim=3, hidden_dims=64)
        x = torch.randn(10, 30)
        z = net.encode(x)
        assert z.shape == (10, 3)

    def test_decode_shape(self):
        """decode returns correct reconstruction shape."""
        from manylatents.algorithms.lightning.networks.gaga_net import GAGANetwork

        net = GAGANetwork(input_dim=30, latent_dim=3, hidden_dims=64)
        z = torch.randn(10, 3)
        x_hat = net.decode(z)
        assert x_hat.shape == (10, 30)

    def test_single_hidden_dim_int(self):
        """A scalar hidden_dims is accepted and produces correct shapes."""
        from manylatents.algorithms.lightning.networks.gaga_net import GAGANetwork

        net = GAGANetwork(input_dim=20, latent_dim=2, hidden_dims=32)
        x = torch.randn(5, 20)
        x_hat, z = net(x)
        assert x_hat.shape == (5, 20)
        assert z.shape == (5, 2)

    def test_batchnorm_and_dropout(self):
        """Network constructs and runs with batchnorm + dropout enabled."""
        from manylatents.algorithms.lightning.networks.gaga_net import GAGANetwork

        net = GAGANetwork(
            input_dim=20,
            latent_dim=2,
            hidden_dims=[16, 8],
            batchnorm=True,
            dropout=0.1,
        )
        x = torch.randn(8, 20)
        x_hat, z = net(x)
        assert x_hat.shape == (8, 20)

    def test_spectral_norm(self):
        """Spectral norm wraps Linear layers (weight_orig present)."""
        from manylatents.algorithms.lightning.networks.gaga_net import GAGANetwork

        net = GAGANetwork(
            input_dim=20, latent_dim=2, hidden_dims=[16], spectral_norm=True
        )
        # Spectral norm replaces weight with weight_orig + weight_v
        first_linear = net.encoder[0]
        assert hasattr(first_linear, "weight_orig")

    def test_activations(self):
        """All supported activations produce valid output."""
        from manylatents.algorithms.lightning.networks.gaga_net import GAGANetwork

        for act in ("relu", "tanh", "sigmoid"):
            net = GAGANetwork(
                input_dim=10, latent_dim=2, hidden_dims=8, activation=act
            )
            x_hat, z = net(torch.randn(4, 10))
            assert not torch.isnan(z).any(), f"NaN in latent with activation={act}"

    def test_gradients_flow(self):
        """Gradients propagate through encode -> decode."""
        from manylatents.algorithms.lightning.networks.gaga_net import GAGANetwork

        net = GAGANetwork(input_dim=10, latent_dim=2, hidden_dims=8)
        x = torch.randn(4, 10)
        x_hat, z = net(x)
        loss = x_hat.sum()
        loss.backward()
        assert net.encoder[0].weight.grad is not None


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------


class TestGAGADistanceLoss:
    """Tests for gaga_distance_loss."""

    def test_returns_scalar(self):
        """Loss is a scalar tensor."""
        from manylatents.algorithms.lightning.networks.gaga_net import (
            gaga_distance_loss,
        )

        z = torch.randn(10, 2)
        gt = torch.nn.functional.pdist(torch.randn(10, 5))
        loss = gaga_distance_loss(z, gt)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_zero_for_matching_distances(self):
        """Loss is zero when latent distances match ground truth exactly."""
        from manylatents.algorithms.lightning.networks.gaga_net import (
            gaga_distance_loss,
        )

        z = torch.randn(8, 3)
        gt = torch.nn.functional.pdist(z)
        loss = gaga_distance_loss(z, gt)
        assert loss.item() < 1e-6

    def test_decay_weighting(self):
        """Non-zero decay produces a different loss value."""
        from manylatents.algorithms.lightning.networks.gaga_net import (
            gaga_distance_loss,
        )

        z = torch.randn(10, 2)
        gt = torch.nn.functional.pdist(torch.randn(10, 5)).abs() + 0.1
        loss_no_decay = gaga_distance_loss(z, gt, dist_mse_decay=0.0)
        loss_decay = gaga_distance_loss(z, gt, dist_mse_decay=1.0)
        # Decay down-weights distant pairs, so the value should differ
        assert not torch.allclose(loss_no_decay, loss_decay)

    def test_gradients(self):
        """Loss propagates gradients to z."""
        from manylatents.algorithms.lightning.networks.gaga_net import (
            gaga_distance_loss,
        )

        z = torch.randn(8, 2, requires_grad=True)
        gt = torch.nn.functional.pdist(torch.randn(8, 5))
        loss = gaga_distance_loss(z, gt)
        loss.backward()
        assert z.grad is not None


class TestGAGAReconstructionLoss:
    """Tests for gaga_reconstruction_loss."""

    def test_returns_scalar(self):
        """Reconstruction loss is a non-negative scalar."""
        from manylatents.algorithms.lightning.networks.gaga_net import (
            gaga_reconstruction_loss,
        )

        x = torch.randn(10, 20)
        x_hat = torch.randn(10, 20)
        loss = gaga_reconstruction_loss(x_hat, x)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_zero_for_perfect_reconstruction(self):
        """Loss is zero when x_hat == x."""
        from manylatents.algorithms.lightning.networks.gaga_net import (
            gaga_reconstruction_loss,
        )

        x = torch.randn(10, 20)
        loss = gaga_reconstruction_loss(x, x)
        assert loss.item() < 1e-7


class TestGAGAAffinityLoss:
    """Tests for gaga_affinity_loss."""

    def _make_stochastic(self, n: int) -> torch.Tensor:
        """Helper: create a random (n, n) row-stochastic matrix."""
        raw = torch.rand(n, n) + 1e-4
        return raw / raw.sum(dim=1, keepdim=True)

    def test_kl_returns_scalar(self):
        """KL loss returns a non-negative scalar."""
        from manylatents.algorithms.lightning.networks.gaga_net import (
            gaga_affinity_loss,
        )

        p = self._make_stochastic(8)
        q = self._make_stochastic(8)
        loss = gaga_affinity_loss(p, q, loss_type="kl")
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_jsd_returns_scalar(self):
        """JSD loss returns a non-negative scalar."""
        from manylatents.algorithms.lightning.networks.gaga_net import (
            gaga_affinity_loss,
        )

        p = self._make_stochastic(8)
        q = self._make_stochastic(8)
        loss = gaga_affinity_loss(p, q, loss_type="jsd")
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_mse_returns_scalar(self):
        """MSE loss returns a non-negative scalar."""
        from manylatents.algorithms.lightning.networks.gaga_net import (
            gaga_affinity_loss,
        )

        p = self._make_stochastic(8)
        q = self._make_stochastic(8)
        loss = gaga_affinity_loss(p, q, loss_type="mse")
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_zero_for_identical_distributions(self):
        """All loss types return near-zero for identical inputs."""
        from manylatents.algorithms.lightning.networks.gaga_net import (
            gaga_affinity_loss,
        )

        p = self._make_stochastic(8)
        for lt in ("kl", "jsd", "mse"):
            loss = gaga_affinity_loss(p, p.clone(), loss_type=lt)
            assert loss.item() < 1e-5, f"{lt} loss should be ~0 for identical inputs"

    def test_jsd_symmetric(self):
        """JSD is symmetric: JSD(p, q) == JSD(q, p)."""
        from manylatents.algorithms.lightning.networks.gaga_net import (
            gaga_affinity_loss,
        )

        p = self._make_stochastic(8)
        q = self._make_stochastic(8)
        loss_pq = gaga_affinity_loss(p, q, loss_type="jsd")
        loss_qp = gaga_affinity_loss(q, p, loss_type="jsd")
        assert torch.allclose(loss_pq, loss_qp, atol=1e-6)

    def test_unknown_loss_type_raises(self):
        """Unknown loss_type raises ValueError."""
        from manylatents.algorithms.lightning.networks.gaga_net import (
            gaga_affinity_loss,
        )

        p = self._make_stochastic(4)
        with pytest.raises(ValueError, match="Unknown loss_type"):
            gaga_affinity_loss(p, p, loss_type="bad")


class TestComputeProbMatrix:
    """Tests for compute_prob_matrix."""

    def test_gaussian_row_stochastic(self):
        """Gaussian kernel produces rows that sum to 1."""
        from manylatents.algorithms.lightning.networks.gaga_net import (
            compute_prob_matrix,
        )

        z = torch.randn(15, 3)
        P = compute_prob_matrix(z, kernel_method="gaussian")
        row_sums = P.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(15), atol=1e-5)

    def test_tstudent_row_stochastic(self):
        """t-Student kernel produces rows that sum to 1."""
        from manylatents.algorithms.lightning.networks.gaga_net import (
            compute_prob_matrix,
        )

        z = torch.randn(15, 3)
        P = compute_prob_matrix(z, kernel_method="tstudent")
        row_sums = P.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(15), atol=1e-5)

    def test_output_shape(self):
        """Output is (N, N) for input (N, d)."""
        from manylatents.algorithms.lightning.networks.gaga_net import (
            compute_prob_matrix,
        )

        z = torch.randn(10, 4)
        P = compute_prob_matrix(z)
        assert P.shape == (10, 10)

    def test_non_negative(self):
        """All entries are non-negative."""
        from manylatents.algorithms.lightning.networks.gaga_net import (
            compute_prob_matrix,
        )

        z = torch.randn(10, 3)
        for km in ("gaussian", "tstudent"):
            P = compute_prob_matrix(z, kernel_method=km)
            assert (P >= 0).all(), f"Negative entry with kernel_method={km}"

    def test_bandwidth_affects_output(self):
        """Different bandwidth values produce different matrices."""
        from manylatents.algorithms.lightning.networks.gaga_net import (
            compute_prob_matrix,
        )

        z = torch.randn(8, 2)
        P1 = compute_prob_matrix(z, bandwidth=0.5)
        P2 = compute_prob_matrix(z, bandwidth=5.0)
        assert not torch.allclose(P1, P2)

    def test_unknown_kernel_raises(self):
        """Unknown kernel_method raises ValueError."""
        from manylatents.algorithms.lightning.networks.gaga_net import (
            compute_prob_matrix,
        )

        with pytest.raises(ValueError, match="Unknown kernel_method"):
            compute_prob_matrix(torch.randn(4, 2), kernel_method="bad")


# ---------------------------------------------------------------------------
# IndexedDatasetWrapper tests
# ---------------------------------------------------------------------------

phate = pytest.importorskip("phate")


class TestIndexedDatasetWrapper:
    """Tests for the IndexedDatasetWrapper."""

    def test_adds_index_to_dict_dataset(self):
        """Wrapper adds 'index' key to dict-returning datasets."""
        from manylatents.algorithms.lightning.gaga import IndexedDatasetWrapper

        class DictDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {"data": torch.randn(5), "metadata": torch.tensor(0)}

        ds = IndexedDatasetWrapper(DictDataset())
        item = ds[3]
        assert "index" in item
        assert item["index"] == 3

    def test_preserves_original_data(self):
        """Original data keys are preserved."""
        from manylatents.algorithms.lightning.gaga import IndexedDatasetWrapper

        class DictDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 5

            def __getitem__(self, idx):
                return {"data": torch.ones(3) * idx, "metadata": torch.tensor(idx)}

        ds = IndexedDatasetWrapper(DictDataset())
        item = ds[2]
        assert "data" in item
        assert "metadata" in item
        assert torch.allclose(item["data"], torch.ones(3) * 2)

    def test_length_unchanged(self):
        """Wrapper doesn't change dataset length."""
        from manylatents.algorithms.lightning.gaga import IndexedDatasetWrapper

        class DictDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 42

            def __getitem__(self, idx):
                return {"data": torch.randn(3)}

        ds = IndexedDatasetWrapper(DictDataset())
        assert len(ds) == 42


# ---------------------------------------------------------------------------
# GAGA LightningModule tests
# ---------------------------------------------------------------------------


def _make_gaga(mode="distance"):
    """Build a minimal GAGA model with a small SwissRoll datamodule."""
    from manylatents.algorithms.lightning.gaga import GAGA
    from manylatents.data.swissroll import SwissRollDataModule

    dm = SwissRollDataModule(
        batch_size=64,
        n_distributions=5,
        n_points_per_distribution=20,  # 100 total points
        noise=0.1,
        manifold_noise=0.1,
    )
    dm.setup()

    network_config = {
        "_target_": "manylatents.algorithms.lightning.networks.gaga_net.GAGANetwork",
        "input_dim": None,  # inferred in setup
        "latent_dim": 2,
        "hidden_dims": [32, 16],
        "activation": "relu",
    }
    optimizer_config = {
        "_target_": "torch.optim.Adam",
        "_partial_": True,
        "lr": 0.001,
    }
    model = GAGA(
        network=network_config,
        optimizer=optimizer_config,
        datamodule=dm,
        mode=mode,
        phate_knn=5,
    )
    return model, dm


class TestGAGALightningModule:
    """Tests for the GAGA LightningModule."""

    def test_instantiation_distance_mode(self):
        """GAGA can be instantiated in distance mode."""
        from manylatents.algorithms.lightning.gaga import GAGA

        model = GAGA(
            network={"_target_": "manylatents.algorithms.lightning.networks.gaga_net.GAGANetwork"},
            optimizer={"_target_": "torch.optim.Adam", "_partial_": True, "lr": 1e-3},
            mode="distance",
        )
        assert model.mode == "distance"
        assert model.automatic_optimization is False
        assert model.network is None

    def test_instantiation_affinity_mode(self):
        """GAGA can be instantiated in affinity mode."""
        from manylatents.algorithms.lightning.gaga import GAGA

        model = GAGA(
            network={"_target_": "manylatents.algorithms.lightning.networks.gaga_net.GAGANetwork"},
            optimizer={"_target_": "torch.optim.Adam", "_partial_": True, "lr": 1e-3},
            mode="affinity",
        )
        assert model.mode == "affinity"
        assert model.automatic_optimization is False

    def test_setup_distance_mode(self):
        """setup() computes PHATE distances and wraps dataset."""
        model, dm = _make_gaga(mode="distance")
        model.setup()

        # Network should be instantiated
        assert model.network is not None
        # Preprocessor should exist
        assert model._preprocessor is not None
        # Ground-truth distance matrix should be set
        assert model._gt_distances is not None
        n = len(dm.train_dataset)  # wrapped now
        assert model._gt_distances.shape == (n, n)
        # Dataset should be wrapped with IndexedDatasetWrapper
        from manylatents.algorithms.lightning.gaga import IndexedDatasetWrapper
        assert isinstance(dm.train_dataset, IndexedDatasetWrapper)

    def test_setup_affinity_mode(self):
        """setup() computes PHATE transition matrix."""
        model, dm = _make_gaga(mode="affinity")
        model.setup()

        assert model.network is not None
        assert model._preprocessor is not None
        assert model._gt_prob_matrix is not None
        n = len(dm.train_dataset)  # wrapped now
        assert model._gt_prob_matrix.shape == (n, n)
        # All training data stored for epoch-end pass
        assert model._all_train_data is not None
        assert model._all_train_data.shape[0] == n

    def test_setup_idempotent(self):
        """Calling setup() twice does not rebuild the network."""
        model, dm = _make_gaga(mode="distance")
        model.setup()
        net_id = id(model.network)
        model.setup()
        assert id(model.network) == net_id

    def test_encode_shape(self):
        """encode() returns correct shape after setup."""
        model, dm = _make_gaga(mode="distance")
        model.setup()

        x = torch.randn(10, 3)  # SwissRoll is 3D
        z = model.encode(x)
        assert z.shape == (10, 2)  # latent_dim=2

    def test_encode_no_grad(self):
        """encode() does not create a computation graph."""
        model, _ = _make_gaga(mode="distance")
        model.setup()
        x = torch.randn(5, 3)
        z = model.encode(x)
        assert not z.requires_grad

    def test_fast_dev_run_distance(self):
        """Distance mode trains without error in fast_dev_run."""
        from lightning.pytorch import Trainer

        model, dm = _make_gaga(mode="distance")
        trainer = Trainer(
            fast_dev_run=True,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
        )
        trainer.fit(model, datamodule=dm)
        # Network should be built after fit
        assert model.network is not None

    def test_fast_dev_run_affinity(self):
        """Affinity mode trains without error in fast_dev_run."""
        from lightning.pytorch import Trainer

        model, dm = _make_gaga(mode="affinity")
        trainer = Trainer(
            fast_dev_run=True,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
        )
        trainer.fit(model, datamodule=dm)
        assert model.network is not None
