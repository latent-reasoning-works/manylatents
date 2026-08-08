"""MIOFlow LightningModule training wrapper.

Manifold Interpolating Optimal-Transport Flows for trajectory inference.
Uses manual optimization to iterate over time intervals within each training step.

Optionally composes a GAGA (Geometry-Aware Generative Autoencoder) encoder:
when ``use_gaga=True``, a GAGA network is pretrained internally (during
``setup()``) against PHATE-embedding pairwise distances, then frozen, and the
ODE flow trains/integrates through its latent space instead of raw ambient
space. Unlike upstream mioflow 2.0 -- which keeps GAGA and MIOFlow as two
classes the caller manually composes (``MIOFlow(gaga_model=<pretrained
Autoencoder>)``) -- manylatents composes them into a single LightningModule
with one Hydra catalog entry and one ``fit()``.

Reference: Huguet et al., arXiv:2206.14928 (2022)
"""

import functools
import logging

import hydra_zen
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from omegaconf import DictConfig
from torch import Tensor

logger = logging.getLogger(__name__)


class MIOFlow(LightningModule):
    """Lightning training wrapper for MIOFlow trajectory inference.

    Trains a Neural ODE velocity field using Optimal Transport losses over
    time-labeled populations. Supports 3-phase training: local pretrain,
    global, and local finetune.

    Args:
        network: Hydra config or instantiated MIOFlowODEFunc. Existing modules
            keep their identity and weights; seed them at construction with
            MIOFlowODEFunc(..., init_seed=42).
        optimizer: Hydra config for optimizer (partial instantiation).
        datamodule: Data module for loading time-labeled data.
        init_seed: Seed before config construction; never resets supplied weights.
        n_local_epochs: Epochs of local (per-interval) pre-training.
        n_global_epochs: Epochs of global (full-trajectory) training.
        n_post_local_epochs: Epochs of local fine-tuning after global.
        lambda_ot: Weight for OT loss.
        lambda_energy: Weight for energy regularisation.
        lambda_density: Weight for density loss.
        energy_time_steps: Sub-steps for energy loss computation.
        sample_size: Points sampled per time step (None = use all).
        n_trajectories: Number of trajectories to generate after training.
        n_bins: Number of time bins for trajectory integration.
        density_top_k: k for the density loss's k-NN hinge penalty.
        density_hinge_value: Hinge threshold for the density loss.
        grad_clip: Max gradient norm (None = no clipping, the historical
            default). Upstream mioflow 2.0 defaults this to 1.0; kept opt-in
            here so existing callers' training dynamics don't silently change.
        use_gaga: If True, pretrain a GAGA encoder/decoder internally during
            ``setup()`` and flow the ODE through its latent space instead of
            raw ambient space. Default False reproduces pre-GAGA behavior
            exactly.
        gaga_latent_dim: GAGA bottleneck dimensionality. Required when
            ``use_gaga=True``.
        gaga_hidden_dims: GAGA encoder hidden layer width(s) (decoder mirrors).
        gaga_activation: GAGA activation function name.
        gaga_encoder_epochs: Phase-1 epochs (distance loss, decoder frozen).
        gaga_decoder_epochs: Phase-2 epochs (reconstruction loss, encoder frozen).
        gaga_dist_weight_phase1: Weight for phase-1 distance loss.
        gaga_recon_weight_phase2: Weight for phase-2 reconstruction loss.
        gaga_lr: Learning rate for GAGA's own (separate) pretraining optimizers.
        gaga_phate_knn: PHATE ``knn`` used to compute GAGA's target distances.
        gaga_phate_t: PHATE ``t`` (diffusion time) used for target distances.
        gaga_phate_n_components: Dimensionality of the PHATE embedding whose
            Euclidean pairwise distances GAGA's encoder is trained to match.
    """

    def __init__(
        self,
        network,
        optimizer,
        datamodule=None,
        init_seed: int = 42,
        # Training phases
        n_local_epochs: int = 0,
        n_global_epochs: int = 100,
        n_post_local_epochs: int = 0,
        # Loss weights
        lambda_ot: float = 1.0,
        lambda_energy: float = 0.01,
        lambda_density: float = 0.0,
        energy_time_steps: int = 10,
        # Data
        sample_size: int | None = None,
        # Output
        n_trajectories: int = 100,
        n_bins: int = 100,
        # Density loss knobs (previously hardcoded)
        density_top_k: int = 5,
        density_hinge_value: float = 0.01,
        # Optimization
        grad_clip: float | None = None,
        # GAGA (optional composed geometric autoencoder)
        use_gaga: bool = False,
        gaga_latent_dim: int | None = None,
        gaga_hidden_dims: list[int] | int = 128,
        gaga_activation: str = "relu",
        gaga_encoder_epochs: int = 50,
        gaga_decoder_epochs: int = 50,
        gaga_dist_weight_phase1: float = 1.0,
        gaga_recon_weight_phase2: float = 1.0,
        gaga_lr: float = 1e-3,
        gaga_phate_knn: int = 5,
        gaga_phate_t: str | int = "auto",
        gaga_phate_n_components: int = 2,
    ):
        super().__init__()
        self.automatic_optimization = False

        if use_gaga and gaga_latent_dim is None:
            raise ValueError("gaga_latent_dim must be set when use_gaga=True.")

        self.datamodule = datamodule
        self.network_config = network
        self.optimizer_config = optimizer
        self.init_seed = init_seed

        # Training phases
        self.n_local_epochs = n_local_epochs
        self.n_global_epochs = n_global_epochs
        self.n_post_local_epochs = n_post_local_epochs

        # Loss weights
        self.lambda_ot = lambda_ot
        self.lambda_energy = lambda_energy
        self.lambda_density = lambda_density
        self.energy_time_steps = energy_time_steps

        # Data
        self.sample_size = sample_size

        # Output
        self.n_trajectories = n_trajectories
        self.n_bins = n_bins

        # Density loss knobs
        self.density_top_k = density_top_k
        self.density_hinge_value = density_hinge_value

        # Optimization
        self.grad_clip = grad_clip

        # GAGA
        self.use_gaga = use_gaga
        self.gaga_latent_dim = gaga_latent_dim
        self.gaga_hidden_dims = gaga_hidden_dims
        self.gaga_activation = gaga_activation
        self.gaga_encoder_epochs = gaga_encoder_epochs
        self.gaga_decoder_epochs = gaga_decoder_epochs
        self.gaga_dist_weight_phase1 = gaga_dist_weight_phase1
        self.gaga_recon_weight_phase2 = gaga_recon_weight_phase2
        self.gaga_lr = gaga_lr
        self.gaga_phate_knn = gaga_phate_knn
        self.gaga_phate_t = gaga_phate_t
        self.gaga_phate_n_components = gaga_phate_n_components

        self.save_hyperparameters(ignore=["datamodule", "network"])
        self.network: nn.Module | None = None
        self.gaga_network: nn.Module | None = None
        self._gaga_preprocessor: nn.Module | None = None
        self._trajectories: Tensor | None = None

    @property
    def total_epochs(self) -> int:
        return self.n_local_epochs + self.n_global_epochs + self.n_post_local_epochs

    def _get_training_mode(self, epoch: int) -> str:
        """Determine training mode based on current epoch."""
        if epoch < self.n_local_epochs:
            return "local"
        elif epoch < self.n_local_epochs + self.n_global_epochs:
            return "global"
        else:
            return "local"

    def setup(self, stage=None):
        """Optionally pretrain GAGA, infer input_dim, then build the ODE network."""
        if self.network is not None:
            return

        needs_ambient_dim = self.use_gaga or (
            isinstance(self.network_config, (dict, DictConfig))
            and self.network_config.get("input_dim") is None
        )
        ambient_dim = None
        if needs_ambient_dim and self.datamodule is not None:
            first_batch = next(iter(self.datamodule.train_dataloader()))
            data = first_batch["data"] if isinstance(first_batch, dict) else first_batch[0]
            ambient_dim = data.shape[1]

        if self.use_gaga:
            if ambient_dim is None:
                raise ValueError(
                    "use_gaga=True requires a datamodule to infer the ambient dimension."
                )
            self._setup_gaga(ambient_dim)

        if isinstance(self.network_config, (dict, DictConfig)):
            if self.network_config.get("input_dim") is None:
                self.network_config["input_dim"] = (
                    self.gaga_latent_dim if self.use_gaga else ambient_dim
                )
        self.configure_model()

    def configure_model(self):
        """Instantiate network from Hydra config."""
        # Lightning calls this hook for every stage, even when setup() returns early.
        if self.network is not None:
            return
        torch.manual_seed(self.init_seed)
        if isinstance(self.network_config, (dict, DictConfig)):
            self.network = hydra_zen.instantiate(self.network_config)
        else:
            self.network = self.network_config
        logger.info(f"MIOFlow network: {self.network.__class__.__name__}")

    def _setup_gaga(self, ambient_dim: int) -> None:
        """Pretrain a GAGA encoder/decoder against PHATE-embedding distances.

        Runs entirely inside ``setup()`` as a self-contained two-phase loop
        with its own optimizers -- not wired into the local/global/local
        Lightning epoch schedule -- so ``n_local_epochs``/``n_global_epochs``/
        ``current_epoch`` semantics are unaffected by ``use_gaga``. Mirrors
        the imperative-loop precedent in ``phase1_align.align_on_snapshot``.
        """
        from .networks.gaga_net import GAGANetwork, Preprocessor

        all_data = []
        for batch in self.datamodule.train_dataloader():
            data = batch["data"] if isinstance(batch, dict) else batch[0]
            all_data.append(data)
        all_data_tensor = torch.cat(all_data, dim=0)

        gt_distances = self._compute_gaga_target_distances(all_data_tensor.numpy())

        mean = all_data_tensor.mean(dim=0)
        std = all_data_tensor.std(dim=0).clamp(min=1e-8)
        dist_std = torch.tensor(float(gt_distances.std()), dtype=torch.float32).clamp(min=1e-8)
        self._gaga_preprocessor = Preprocessor(mean=mean, std=std, dist_std=dist_std)

        torch.manual_seed(self.init_seed)
        self.gaga_network = GAGANetwork(
            input_dim=ambient_dim,
            latent_dim=self.gaga_latent_dim,
            hidden_dims=self.gaga_hidden_dims,
            activation=self.gaga_activation,
        )

        self._train_gaga_two_phase(all_data_tensor, gt_distances)
        logger.info(
            f"GAGA pretrained: latent_dim={self.gaga_latent_dim}, "
            f"encoder_epochs={self.gaga_encoder_epochs}, "
            f"decoder_epochs={self.gaga_decoder_epochs}"
        )

    def _compute_gaga_target_distances(self, data_np) -> np.ndarray:
        """Euclidean pairwise distances in a low-dimensional PHATE embedding.

        Matches upstream mioflow 2.0's ``fit_gaga`` target (distance in
        PHATE-embedding coordinates), not the diffusion-potential distance
        used by manylatents' pre-removal GAGA module.
        """
        import phate
        from scipy.spatial.distance import cdist

        phate_op = phate.PHATE(
            n_components=self.gaga_phate_n_components,
            knn=self.gaga_phate_knn,
            t=self.gaga_phate_t,
            verbose=0,
        )
        phate_embedding = phate_op.fit_transform(data_np)
        return cdist(phate_embedding, phate_embedding, metric="euclidean").astype("float32")

    def _train_gaga_two_phase(self, data: Tensor, gt_distances: np.ndarray) -> None:
        """Two-phase GAGA pretraining; frozen thereafter.

        Phase 1 optimizes only the encoder against the distance-preservation
        loss (decoder frozen); phase 2 optimizes only the decoder against
        reconstruction (encoder frozen) -- mirroring upstream's
        ``train_gaga_two_phase``. GAGA is frozen afterwards and never
        jointly fine-tuned with the ODE flow, matching upstream's frozen-GAE
        staging.
        """
        from .networks.gaga_net import gaga_distance_loss, gaga_reconstruction_loss

        device = self.device
        self.gaga_network = self.gaga_network.to(device)
        self._gaga_preprocessor = self._gaga_preprocessor.to(device)
        data = data.to(device)
        x_norm = self._gaga_preprocessor.normalize(data)

        triu_idx = np.triu_indices(gt_distances.shape[0], k=1)
        gt_upper = torch.tensor(gt_distances[triu_idx], dtype=x_norm.dtype, device=device)
        gt_upper = self._gaga_preprocessor.normalize_dist(gt_upper)

        def _set_requires_grad(module: nn.Module, flag: bool) -> None:
            for p in module.parameters():
                p.requires_grad_(flag)

        self.gaga_network.train()

        # Phase 1: distance-preserving encoder, decoder frozen.
        _set_requires_grad(self.gaga_network.encoder, True)
        _set_requires_grad(self.gaga_network.decoder, False)
        opt1 = torch.optim.Adam(self.gaga_network.encoder.parameters(), lr=self.gaga_lr)
        for _ in range(self.gaga_encoder_epochs):
            opt1.zero_grad()
            z = self.gaga_network.encode(x_norm)
            loss = self.gaga_dist_weight_phase1 * gaga_distance_loss(z, gt_upper)
            loss.backward()
            opt1.step()

        # Phase 2: reconstruction decoder, encoder frozen.
        _set_requires_grad(self.gaga_network.encoder, False)
        _set_requires_grad(self.gaga_network.decoder, True)
        opt2 = torch.optim.Adam(self.gaga_network.decoder.parameters(), lr=self.gaga_lr)
        for _ in range(self.gaga_decoder_epochs):
            opt2.zero_grad()
            with torch.no_grad():
                z = self.gaga_network.encode(x_norm)
            x_hat = self.gaga_network.decode(z)
            loss = self.gaga_recon_weight_phase2 * gaga_reconstruction_loss(x_hat, x_norm)
            loss.backward()
            opt2.step()

        # Frozen thereafter -- never jointly fine-tuned with the ODE flow.
        _set_requires_grad(self.gaga_network, False)
        self.gaga_network.eval()

    def _maybe_gaga_encode(self, x: Tensor) -> Tensor:
        """Encode through the frozen GAGA network when enabled; identity otherwise."""
        if not self.use_gaga:
            return x
        return self._gaga_encode(x)

    def _gaga_encode(self, x: Tensor) -> Tensor:
        assert self.gaga_network is not None, "GAGA network not configured. Call setup() first."
        with torch.no_grad():
            x_norm = self._gaga_preprocessor.normalize(x)
            return self.gaga_network.encode(x_norm)

    def _gaga_decode(self, z: Tensor) -> Tensor:
        assert self.gaga_network is not None, "GAGA network not configured. Call setup() first."
        with torch.no_grad():
            x_norm = self.gaga_network.decode(z)
            return self._gaga_preprocessor.unnormalize(x_norm)

    def _group_by_time(self, batch: dict) -> list[tuple[Tensor, float]]:
        """Group batch data by per-sample timepoint into a sorted list of (X_t, t).

        Three keys, read in this order:

        - ``"time"`` — the dedicated per-cell timepoint channel that
          ``api.run(time=...)`` threads through ``PrecomputedDataModule`` into
          ``InMemoryDataset`` (``data/precomputed_dataset.py:55``), and the key
          ``Cflows`` reads (``cflows.py:225``). MIOFlow predates it and never
          learned about it, so ``api.run(algorithms={"lightning": "mioflow"},
          time=t)`` raised ``KeyError`` on a batch — keys ``['data',
          'embeddings', 'time']`` — that carried the timepoints all along.
        - ``"labels"`` (plural) — MIOFlow's own prototype datamodules.
        - ``"label"`` (singular) — the manyLatents op-contract key; geomancer's
          ``pipeline/mioflow.py:97`` (on geomancer ``main``) still emits it under that key.

        ``"time"`` wins because it is the only *unambiguous* one — ``"label"`` is
        also the colouring/metric channel. Measured before this change: a batch
        carrying a 2-class cell-type ``label`` beside a 4-timepoint ``time`` grouped
        into 2 "timepoints" and trained a flow between cell types. A well-formed
        wrong answer, not a crash.
        """
        if isinstance(batch, dict):
            data = batch["data"]
            # Two `is None` checks rather than one `or` chain, and the reason is stronger
            # than "an all-zero column is falsy": these are TENSORS, and `bool()` on one with
            # more than one element RAISES — `RuntimeError: Boolean value of Tensor with more
            # than one value is ambiguous`. So `batch.get("time") or batch.get("labels")` does
            # not silently fall through to the wrong key, it crashes the step. Measured: the
            # `or` form fails 9 of 19 tests in this file. Loud rather than silent, but a
            # timepoint column is exactly the shape that trips it, so it never gets to be
            # either.
            labels = batch.get("time")
            if labels is None:
                labels = batch.get("labels", batch.get("label"))
            if labels is None:
                raise KeyError(
                    "MIOFlow requires a per-sample timepoint in the batch under "
                    "'time' (pass `time=...` to api.run / the datamodule), "
                    f"'labels' or 'label'; got keys {sorted(batch.keys())}."
                )
        else:
            data, labels = batch[0], batch[1]
        # A CONTINUOUS TIME CHANNEL DEGENERATES HERE, and reading `"time"` makes that newly
        # reachable — so it is named rather than left to be found. Grouping is `torch.unique`, so
        # `time=linspace(0, 1, n)` yields n singleton groups: MEASURED, a 40-cell run produced 40
        # groups of one, trained to completion and returned finite embeddings with no warning.
        # The OT loss between one-point distributions is degenerate and the flow it fits is not
        # the one the caller meant. Not refused here, because a legitimate two-timepoint run is
        # indistinguishable from a degenerate one by count alone and `_group_by_time` is on the
        # training hot path; the honest place for that judgement is the caller that knows whether
        # its clock is categorical. Recorded so the next person does not re-derive it.
        unique_times = torch.unique(labels, sorted=True)
        groups = []
        for t in unique_times:
            mask = labels == t
            groups.append((data[mask], t.item()))
        return groups

    def _local_step(self, time_groups: list[tuple[Tensor, float]]) -> dict[str, Tensor]:
        """Train on each consecutive time interval independently."""
        from torchdiffeq import odeint

        from .networks.mioflow_net import (
            mioflow_density_loss,
            mioflow_energy_loss,
            mioflow_ot_loss,
        )

        device = self.device
        total_loss = torch.tensor(0.0, device=device)
        ot_sum = torch.tensor(0.0, device=device)
        energy_sum = torch.tensor(0.0, device=device)
        density_sum = torch.tensor(0.0, device=device)
        n_intervals = 0

        for i in range(len(time_groups) - 1):
            X_start, t_start = time_groups[i]
            X_end, t_end = time_groups[i + 1]
            X_start = X_start.to(device)
            X_end = X_end.to(device)

            # Subsample if needed
            if self.sample_size is not None:
                n = min(X_start.size(0), X_end.size(0), self.sample_size)
                X_start = X_start[torch.randperm(X_start.size(0))[:n]]
                X_end = X_end[torch.randperm(X_end.size(0))[:n]]

            X_start = self._maybe_gaga_encode(X_start)
            X_end = self._maybe_gaga_encode(X_end)

            t_interval = torch.tensor([t_start, t_end], device=device, dtype=torch.float32)
            if hasattr(self.network, "reset_momentum"):
                self.network.reset_momentum()
            X_pred = odeint(self.network, X_start, t_interval)[1]

            interval_loss = torch.tensor(0.0, device=device)

            if self.lambda_ot > 0:
                ot_v = mioflow_ot_loss(X_pred, X_end)
                interval_loss = interval_loss + self.lambda_ot * ot_v
                ot_sum = ot_sum + ot_v.detach()

            if self.lambda_density > 0:
                den_v = mioflow_density_loss(
                    X_pred, X_end, top_k=self.density_top_k, hinge_value=self.density_hinge_value
                )
                interval_loss = interval_loss + self.lambda_density * den_v
                density_sum = density_sum + den_v.detach()

            if self.lambda_energy > 0:
                e_t = torch.linspace(t_start, t_end, self.energy_time_steps, device=device)
                eng_v = mioflow_energy_loss(self.network, X_start, e_t)
                interval_loss = interval_loss + self.lambda_energy * eng_v
                energy_sum = energy_sum + eng_v.detach()

            total_loss = total_loss + interval_loss
            n_intervals += 1

        if n_intervals > 0:
            total_loss = total_loss / n_intervals

        return {
            "loss": total_loss,
            "ot_loss": ot_sum / max(n_intervals, 1),
            "energy_loss": energy_sum / max(n_intervals, 1),
            "density_loss": density_sum / max(n_intervals, 1),
        }

    def _global_step(self, time_groups: list[tuple[Tensor, float]]) -> dict[str, Tensor]:
        """Train on full trajectory end-to-end."""
        from torchdiffeq import odeint

        from .networks.mioflow_net import (
            mioflow_density_loss,
            mioflow_energy_loss,
            mioflow_ot_loss,
        )

        device = self.device
        X_0 = time_groups[0][0].to(device)
        times = [t for _, t in time_groups]
        t_seq = torch.tensor(times, device=device, dtype=torch.float32)

        if self.sample_size is not None and X_0.size(0) > self.sample_size:
            idx = torch.randperm(X_0.size(0))[: self.sample_size]
            X_0 = X_0[idx]

        X_0 = self._maybe_gaga_encode(X_0)

        if hasattr(self.network, "reset_momentum"):
            self.network.reset_momentum()
        trajectory = odeint(self.network, X_0, t_seq)

        total_loss = torch.tensor(0.0, device=device)
        ot_sum = torch.tensor(0.0, device=device)
        density_sum = torch.tensor(0.0, device=device)

        for i in range(1, len(t_seq)):
            X_pred = trajectory[i]
            X_true = time_groups[i][0].to(device)

            if self.sample_size is not None and X_true.size(0) > self.sample_size:
                X_true = X_true[torch.randperm(X_true.size(0))[: self.sample_size]]

            X_true = self._maybe_gaga_encode(X_true)

            if self.lambda_ot > 0:
                ot_v = mioflow_ot_loss(X_pred, X_true)
                total_loss = total_loss + self.lambda_ot * ot_v
                ot_sum = ot_sum + ot_v.detach()

            if self.lambda_density > 0:
                den_v = mioflow_density_loss(
                    X_pred, X_true, top_k=self.density_top_k, hinge_value=self.density_hinge_value
                )
                total_loss = total_loss + self.lambda_density * den_v
                density_sum = density_sum + den_v.detach()

        energy_v = torch.tensor(0.0, device=device)
        if self.lambda_energy > 0:
            e_t = torch.linspace(times[0], times[-1], self.energy_time_steps, device=device)
            energy_v = mioflow_energy_loss(self.network, X_0, e_t)
            total_loss = total_loss + self.lambda_energy * energy_v

        return {
            "loss": total_loss,
            "ot_loss": ot_sum / max(len(t_seq) - 1, 1),
            "energy_loss": energy_v.detach(),
            "density_loss": density_sum / max(len(t_seq) - 1, 1),
        }

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        mode = self._get_training_mode(self.current_epoch)
        time_groups = self._group_by_time(batch)

        if len(time_groups) < 2:
            return  # Need at least 2 time points

        if mode == "local":
            result = self._local_step(time_groups)
        else:
            result = self._global_step(time_groups)

        opt.zero_grad()
        self.manual_backward(result["loss"])
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        opt.step()

        self.log("train_loss", result["loss"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_ot_loss", result["ot_loss"], on_step=False, on_epoch=True)
        self.log("train_energy_loss", result["energy_loss"], on_step=False, on_epoch=True)
        self.log("train_density_loss", result["density_loss"], on_step=False, on_epoch=True)
        self.log("training_mode", float(0 if mode == "local" else 1), on_step=False, on_epoch=True)

    def on_train_end(self):
        """Generate trajectories after training completes."""
        self._generate_trajectories()

    def _generate_trajectories(self):
        """Integrate n_trajectories paths over n_bins time steps."""
        from torchdiffeq import odeint

        if self.datamodule is None:
            return

        # Get all data grouped by time
        all_data = []
        for batch in self.datamodule.train_dataloader():
            groups = self._group_by_time(batch)
            all_data.extend(groups)

        if not all_data:
            return

        # Sort by time and deduplicate
        all_data.sort(key=lambda x: x[1])
        times = sorted(set(t for _, t in all_data))

        # Get initial conditions from earliest time
        X_0 = all_data[0][0].to(self.device)
        n = min(self.n_trajectories, X_0.size(0))
        idx = torch.randperm(X_0.size(0))[:n]
        X_0_sample = X_0[idx]

        t_bins = torch.linspace(min(times), max(times), self.n_bins, device=self.device)

        self.network.eval()
        with torch.no_grad():
            X_0_latent = self._maybe_gaga_encode(X_0_sample)
            if hasattr(self.network, "reset_momentum"):
                self.network.reset_momentum()
            trajectory = odeint(self.network, X_0_latent, t_bins)
            if self.use_gaga:
                n_bins_, n_traj_, d_latent = trajectory.shape
                flat_ambient = self._gaga_decode(trajectory.reshape(-1, d_latent))
                trajectory = flat_ambient.reshape(n_bins_, n_traj_, -1)
            self._trajectories = trajectory
        self.network.train()
        logger.info(f"Trajectories generated: shape={self._trajectories.shape}")

    def encode(self, x: Tensor, t_start: float | None = None, t_end: float | None = None) -> Tensor:
        """Integrate x from t_start to t_end, return endpoint positions.

        This produces (n, d) embeddings compatible with the pipeline. When
        ``use_gaga=True``, ``x`` is first encoded into GAGA latent space and
        the returned endpoint stays in that latent space (``d ==
        gaga_latent_dim``), matching upstream mioflow 2.0 where the flow
        always operates on an already-reduced space when GAGA is supplied.

        Args:
            x: Ambient-space input points.
            t_start: Integration start time. Defaults (with ``t_end``) to the
                min time label inferred from ``self.datamodule``.
            t_end: Integration end time. Defaults to the max time label.
        """
        from torchdiffeq import odeint

        assert self.network is not None, "Network not configured. Call setup() first."

        if t_start is not None and t_end is not None:
            t_span = torch.tensor([t_start, t_end], device=x.device, dtype=torch.float32)
        else:
            # Get time range from data if available
            t_span = torch.tensor([0.0, 1.0], device=x.device, dtype=torch.float32)
            if self.datamodule is not None:
                try:
                    batch = next(iter(self.datamodule.train_dataloader()))
                    groups = self._group_by_time(batch)
                    if len(groups) >= 2:
                        times = [t for _, t in groups]
                        t_span = torch.tensor(
                            [min(times), max(times)], device=x.device, dtype=torch.float32
                        )
                except StopIteration:
                    pass

        self.network.eval()
        with torch.no_grad():
            z = self._maybe_gaga_encode(x)
            if hasattr(self.network, "reset_momentum"):
                self.network.reset_momentum()
            trajectory = odeint(self.network, z, t_span)
        return trajectory[-1]  # endpoint positions

    @property
    def trajectories(self) -> Tensor | None:
        """Full trajectories (n_bins, n_traj, d) if generated."""
        return self._trajectories

    def test_step(self, batch, batch_idx):
        time_groups = self._group_by_time(batch)
        if len(time_groups) < 2:
            return
        result = self._global_step(time_groups)
        self.log("test_loss", result["loss"], prog_bar=True, on_epoch=True)
        return result

    def configure_optimizers(self):
        """Instantiate optimizer."""
        if isinstance(self.optimizer_config, functools.partial):
            optimizer = self.optimizer_config(self.parameters())
        else:
            optimizer_partial = hydra_zen.instantiate(self.optimizer_config)
            optimizer = optimizer_partial(self.parameters())
        return optimizer
