"""GAGA LightningModule training wrapper.

Geometry-Aware Generative Autoencoder for dimensionality reduction that
preserves pairwise distances (distance mode) or affinity structure
(affinity mode) computed from PHATE geometry.

Uses manual optimization. In distance mode, each training step computes
distance-preserving and reconstruction losses on batch submatrices. In
affinity mode, a full-dataset forward pass is performed at the end of
each epoch to compare the predicted probability matrix against the
PHATE-derived ground-truth transition matrix.

Reference: Huguet et al., "Geodesic Sinkhorn for Fast and Accurate
Optimal Transport on Manifolds" (2023)
"""

from __future__ import annotations

import functools
import logging

import hydra_zen
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from omegaconf import DictConfig
from torch import Tensor

logger = logging.getLogger(__name__)


class IndexedDatasetWrapper(torch.utils.data.Dataset):
    """Wraps a dataset to include sample indices in the batch dict.

    When the underlying dataset returns a ``dict``, the wrapper adds an
    ``"index"`` key.  Otherwise it wraps the first element as
    ``{"data": item[0], "index": idx}``.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, dict):
            item["index"] = idx
        else:
            item = {"data": item[0], "index": idx}
        return item


class GAGA(LightningModule):
    """Lightning training wrapper for GAGA dimensionality reduction.

    Supports two training modes:

    * **distance** -- Preserves pairwise PHATE diffusion-potential distances
      in the latent space (AEDist).
    * **affinity** -- Preserves the PHATE row-stochastic diffusion operator
      via a kernel-based probability matrix in the latent space (AEProb).

    Args:
        network: Hydra config or instantiated GAGANetwork.
        optimizer: Hydra config for optimizer (partial instantiation).
        datamodule: Data module for loading data.
        init_seed: Random seed for weight initialization.
        mode: ``"distance"`` or ``"affinity"``.
        phate_knn: Number of nearest neighbours for PHATE.
        phate_t: Diffusion time for PHATE (``"auto"`` or int).
        phate_n_landmark: Number of landmarks for PHATE (affinity mode).
        dist_weight: Weight for the distance loss (distance mode).
        reconstr_weight: Weight for reconstruction loss (distance mode).
        cycle_weight: Weight for cycle-consistency loss (distance mode).
        dist_mse_decay: Exponential decay factor for distance weighting.
        affinity_weight: Weight for the affinity loss (affinity mode).
        affinity_reconstr_weight: Weight for reconstruction loss (affinity mode).
        loss_type: Affinity divergence: ``"kl"``, ``"jsd"``, or ``"mse"``.
        kernel_method: Kernel for predicted probability: ``"gaussian"`` or ``"tstudent"``.
        kernel_alpha: Exponent for Gaussian kernel.
        kernel_bandwidth: Length-scale for Gaussian kernel.
    """

    def __init__(
        self,
        network,
        optimizer,
        datamodule=None,
        init_seed: int = 42,
        mode: str = "distance",
        # PHATE parameters
        phate_knn: int = 5,
        phate_t: str | int = "auto",
        phate_n_landmark: int = 5000,
        # Distance mode parameters
        dist_weight: float = 0.9,
        reconstr_weight: float = 0.1,
        cycle_weight: float = 0.0,
        dist_mse_decay: float = 0.0,
        # Affinity mode parameters
        affinity_weight: float = 1.0,
        affinity_reconstr_weight: float = 0.1,
        loss_type: str = "kl",
        kernel_method: str = "gaussian",
        kernel_alpha: float = 1.0,
        kernel_bandwidth: float = 1.0,
    ):
        super().__init__()
        self.automatic_optimization = False

        self.datamodule = datamodule
        self.network_config = network
        self.optimizer_config = optimizer
        self.init_seed = init_seed
        self.mode = mode

        # PHATE
        self.phate_knn = phate_knn
        self.phate_t = phate_t
        self.phate_n_landmark = phate_n_landmark

        # Distance mode
        self.dist_weight = dist_weight
        self.reconstr_weight = reconstr_weight
        self.cycle_weight = cycle_weight
        self.dist_mse_decay = dist_mse_decay

        # Affinity mode
        self.affinity_weight = affinity_weight
        self.affinity_reconstr_weight = affinity_reconstr_weight
        self.loss_type = loss_type
        self.kernel_method = kernel_method
        self.kernel_alpha = kernel_alpha
        self.kernel_bandwidth = kernel_bandwidth

        self.save_hyperparameters(ignore=["datamodule", "network"])

        # Network is instantiated lazily in setup()
        self.network: nn.Module | None = None

        # Preprocessor (separate nn.Module so it moves with the model)
        self._preprocessor: nn.Module | None = None

        # Ground-truth targets (plain numpy arrays -- can be large)
        self._gt_distances: np.ndarray | None = None
        self._gt_prob_matrix: np.ndarray | None = None

        # Full training data tensor (used for affinity epoch-end forward)
        self._all_train_data: Tensor | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage=None):
        """Compute PHATE geometry, wrap dataset, and build network."""
        if self.network is not None:
            return  # idempotent guard

        # Collect ALL training data into a single tensor
        all_data = []
        for batch in self.datamodule.train_dataloader():
            data = batch["data"] if isinstance(batch, dict) else batch[0]
            all_data.append(data)
        all_data_tensor = torch.cat(all_data, dim=0)
        data_np = all_data_tensor.numpy()

        input_dim = data_np.shape[1]

        # Compute PHATE geometry targets
        if self.mode == "distance":
            self._setup_distance_mode(data_np)
        elif self.mode == "affinity":
            self._setup_affinity_mode(data_np)
        else:
            raise ValueError(
                f"Unknown mode: {self.mode!r}. Use 'distance' or 'affinity'."
            )

        # Dataset statistics for Preprocessor
        mean = all_data_tensor.mean(dim=0)
        std = all_data_tensor.std(dim=0).clamp(min=1e-8)
        dist_std = (
            torch.tensor(self._gt_distances.std(), dtype=torch.float32)
            if self._gt_distances is not None
            else torch.tensor(1.0)
        )

        # Build the Preprocessor
        from .networks.gaga_net import Preprocessor

        self._preprocessor = Preprocessor(mean=mean, std=std, dist_std=dist_std)

        # Wrap the train dataset with IndexedDatasetWrapper
        self.datamodule.train_dataset = IndexedDatasetWrapper(
            self.datamodule.train_dataset
        )

        # Store all training data for affinity epoch-end pass
        if self.mode == "affinity":
            self._all_train_data = all_data_tensor

        # Infer input_dim from data if needed and instantiate network
        if isinstance(self.network_config, (dict, DictConfig)):
            if self.network_config.get("input_dim") is None:
                self.network_config["input_dim"] = input_dim
        self.configure_model()

    def _setup_distance_mode(self, data_np: np.ndarray):
        """Run PHATE to compute diffusion-potential pairwise distances."""
        import phate
        from scipy.spatial.distance import cdist

        n_samples, input_dim = data_np.shape
        logger.info(
            f"GAGA distance mode: running PHATE on {n_samples} samples "
            f"(knn={self.phate_knn}, t={self.phate_t})"
        )

        phate_op = phate.PHATE(
            n_components=input_dim,
            knn=self.phate_knn,
            t=self.phate_t,
            verbose=0,
        )
        diff_potential = phate_op.fit_transform(data_np)

        # NxN pairwise distance matrix from diffusion potential
        dist_matrix = cdist(diff_potential, diff_potential, metric="euclidean")
        self._gt_distances = dist_matrix.astype(np.float32)
        logger.info(
            f"GAGA distance mode: distance matrix shape={dist_matrix.shape}, "
            f"dist_std={dist_matrix.std():.4f}"
        )

    def _setup_affinity_mode(self, data_np: np.ndarray):
        """Run PHATE to extract the row-stochastic diffusion operator."""
        import phate

        n_samples = data_np.shape[0]
        logger.info(
            f"GAGA affinity mode: running PHATE on {n_samples} samples "
            f"(knn={self.phate_knn}, t={self.phate_t}, "
            f"n_landmark={self.phate_n_landmark})"
        )

        phate_op = phate.PHATE(
            n_components=2,
            knn=self.phate_knn,
            t=self.phate_t,
            n_landmark=self.phate_n_landmark,
            verbose=0,
        )
        phate_op.fit(data_np)

        # Extract the row-stochastic diffusion operator
        diff_op = phate_op.graph.diff_op
        # Convert sparse matrix to dense numpy if needed
        if hasattr(diff_op, "toarray"):
            diff_op = diff_op.toarray()
        self._gt_prob_matrix = np.array(diff_op, dtype=np.float32)
        logger.info(
            f"GAGA affinity mode: transition matrix shape="
            f"{self._gt_prob_matrix.shape}"
        )

    def configure_model(self):
        """Instantiate the GAGANetwork from Hydra config."""
        torch.manual_seed(self.init_seed)
        if isinstance(self.network_config, (dict, DictConfig)):
            self.network = hydra_zen.instantiate(self.network_config)
        else:
            self.network = self.network_config
        logger.info(f"GAGA network: {self.network.__class__.__name__}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        """Dispatch to distance or affinity training step."""
        if self.mode == "distance":
            self._distance_step(batch)
        # In affinity mode, training_step is a no-op; loss computed at epoch end
        return None

    def _distance_step(self, batch):
        """Distance-preserving training step with manual optimization."""
        from .networks.gaga_net import (
            gaga_distance_loss,
            gaga_reconstruction_loss,
        )

        opt = self.optimizers()
        x = batch["data"]
        indices = batch["index"]

        # Slice NxN distance matrix for this batch -> (B, B) submatrix
        idx_np = indices.cpu().numpy()
        gt_dist_batch = self._gt_distances[idx_np][:, idx_np]
        # Upper-triangular vector to match pdist output
        triu_idx = np.triu_indices(len(idx_np), k=1)
        gt_upper = torch.tensor(
            gt_dist_batch[triu_idx],
            dtype=x.dtype,
            device=x.device,
        )
        # Normalize distances
        gt_upper = gt_upper / self._preprocessor.dist_std

        # Forward pass
        x_norm = self._preprocessor.normalize(x)
        x_hat_norm, z = self.network(x_norm)

        # Losses
        dist_loss = gaga_distance_loss(z, gt_upper, self.dist_mse_decay)
        reconstr_loss = gaga_reconstruction_loss(x_hat_norm, x_norm)
        total_loss = self.dist_weight * dist_loss + self.reconstr_weight * reconstr_loss

        # Cycle consistency loss
        if self.cycle_weight > 0:
            z2 = self.network.encode(x_hat_norm)
            cycle_loss = F.mse_loss(z, z2)
            total_loss = total_loss + self.cycle_weight * cycle_loss
            self.log(
                "train_cycle_loss",
                cycle_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

        opt.zero_grad()
        self.manual_backward(total_loss)
        opt.step()

        self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "train_dist_loss",
            dist_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_reconstr_loss",
            reconstr_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

    def on_train_epoch_end(self):
        """For affinity mode: full-dataset forward pass + loss + backward."""
        if self.mode != "affinity":
            return

        from .networks.gaga_net import (
            compute_prob_matrix,
            gaga_affinity_loss,
            gaga_reconstruction_loss,
        )

        opt = self.optimizers()

        # Full-dataset forward pass
        x = self._all_train_data.to(self.device)
        x_norm = self._preprocessor.normalize(x)
        x_hat_norm, z = self.network(x_norm)

        # Predicted probability matrix from latent embeddings
        pred_prob = compute_prob_matrix(
            z, self.kernel_method, self.kernel_alpha, self.kernel_bandwidth
        )

        # Ground truth (full matrix, already row-stochastic)
        gt_prob = torch.tensor(
            self._gt_prob_matrix,
            dtype=z.dtype,
            device=z.device,
        )

        # Compute losses
        aff_loss = gaga_affinity_loss(pred_prob, gt_prob, self.loss_type)
        reconstr_loss = gaga_reconstruction_loss(x_hat_norm, x_norm)
        total_loss = (
            self.affinity_weight * aff_loss
            + self.affinity_reconstr_weight * reconstr_loss
        )

        opt.zero_grad()
        self.manual_backward(total_loss)
        opt.step()

        self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "train_affinity_loss",
            aff_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_reconstr_loss",
            reconstr_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def encode(self, x: Tensor) -> Tensor:
        """Encode input data to the latent space.

        Returns ``(n, latent_dim)`` embeddings for the evaluation pipeline.
        """
        assert self.network is not None, "Network not configured. Call setup() first."
        self.network.eval()
        with torch.no_grad():
            x_norm = self._preprocessor.normalize(x)
            z = self.network.encode(x_norm)
        return z

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Instantiate optimizer from Hydra config or partial."""
        if isinstance(self.optimizer_config, functools.partial):
            return self.optimizer_config(self.parameters())
        else:
            optimizer_partial = hydra_zen.instantiate(self.optimizer_config)
            return optimizer_partial(self.parameters())
