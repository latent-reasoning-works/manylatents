"""MIOFlow LightningModule training wrapper.

Manifold Interpolating Optimal-Transport Flows for trajectory inference.
Uses manual optimization to iterate over time intervals within each training step.

An optional encoder config is instantiated in ``configure_model()`` alongside
its preprocessing buffers and the flow. The encoder supplies fixed coordinates
under ``no_grad()``; GAGA is one choice, with optional two-phase pretraining in
``on_fit_start()``. Completed pretraining is persisted and skipped when resuming.

Config-based checkpoints restore without a datamodule or constructor overrides.
For already-instantiated networks/encoders, callers must supply the corresponding
``network=`` / ``encoder=`` object to ``load_from_checkpoint``. Such objects do
not carry a general, serializable architecture description.

Reference: Huguet et al., arXiv:2206.14928 (2022)
"""

import functools
import logging
from copy import deepcopy

import hydra_zen
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from .networks.network import HasDecode, HasEncode

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
        encoder: Optional Hydra network config. Must instantiate an nn.Module
            satisfying HasEncode: encode(x) and a positive integer latent_dim.
            decode is optional: flow training and endpoint embeddings only need
            encoding. Generated trajectories are decoded when decode is available,
            otherwise they remain in latent space. After configure_model(),
            supports_ambient_trajectories reports this without running the flow.
            An instantiated encoder is
            also accepted (supply it again when restoring).
        encoder_pretraining: "none" uses fixed encoder weights and identity
            preprocessing; "gaga" fits PHATE distances and reconstruction, requiring
            GAGANetwork's encoder/decoder structure. Defaults to "gaga" for the
            legacy use_gaga shortcut and "none" otherwise.
        ambient_dim: Input feature count; inferred from encoder config or data
            if omitted, then saved for checkpoint construction.
        latent_dim: Encoder output dimension, inferred from its contract and saved.
        use_gaga: Backward-compatible shortcut for a GAGANetwork encoder config.
        gaga_latent_dim: Bottleneck dimension for the use_gaga shortcut.
        gaga_hidden_dims: Hidden widths for the use_gaga shortcut.
        gaga_activation: Activation for the use_gaga shortcut.
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
        *,
        encoder=None,
        encoder_pretraining: str | None = None,
        ambient_dim: int | None = None,
        latent_dim: int | None = None,
    ):
        super().__init__()
        self.automatic_optimization = False

        if use_gaga and encoder is None and gaga_latent_dim is None:
            raise ValueError("gaga_latent_dim must be set when use_gaga=True.")

        self.datamodule = datamodule
        self.network_config = (
            deepcopy(network) if isinstance(network, (dict, DictConfig)) else network
        )
        if encoder is None and use_gaga:
            encoder = {
                "_target_": "manylatents.algorithms.lightning.networks.gaga_net.GAGANetwork",
                "input_dim": ambient_dim,
                "latent_dim": gaga_latent_dim,
                "hidden_dims": gaga_hidden_dims,
                "activation": gaga_activation,
            }
        self.encoder_config = (
            deepcopy(encoder) if isinstance(encoder, (dict, DictConfig)) else encoder
        )
        self.encoder_pretraining = (
            ("gaga" if use_gaga else "none")
            if encoder_pretraining is None else encoder_pretraining
        )
        if self.encoder_pretraining not in ("none", "gaga"):
            raise ValueError("encoder_pretraining must be 'none' or 'gaga'.")
        if self.encoder_pretraining == "gaga" and encoder is None:
            raise ValueError("GAGA pretraining requires an encoder config.")
        self.ambient_dim = ambient_dim
        self.latent_dim = latent_dim
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

        self.save_hyperparameters(ignore=["datamodule", "network", "encoder"])
        self.network: nn.Module | None = None
        self.encoder: nn.Module | None = None
        self.preprocessor: nn.Module | None = None
        self.register_buffer("_encoder_fitted", torch.tensor(self.encoder_pretraining == "none"))
        self.register_buffer("_time_span", torch.tensor([0.0, 1.0]))
        self.register_buffer("_time_span_fitted", torch.tensor(False))
        self._trajectories: Tensor | None = None

    @property
    def supports_ambient_trajectories(self) -> bool:
        """Whether trajectories use ambient coordinates, after configure_model().

        Without an encoder the flow already operates in ambient space. With
        one, only a callable decode is required; no training or inference runs.
        """
        if self.network is None:
            raise RuntimeError("Call configure_model() before checking trajectory capabilities.")
        return self.encoder is None or (
            isinstance(self.encoder, HasDecode) and callable(self.encoder.decode)
        )

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
        """Construct only; Lightning does not call setup during checkpoint loading."""
        self.configure_model()

    def configure_model(self):
        """Construct both networks and shaped buffers, once, without fitting data.

        Resolved configs and dimensions are saved explicitly despite the general
        exclusion of live network objects from hyperparameters.
        """
        if self.network is not None:
            return
        from .networks.gaga_net import GAGANetwork, Preprocessor

        torch.manual_seed(self.init_seed)
        if self.encoder_config is not None:
            cfg = self.encoder_config
            configured_dim = (
                cfg.get("input_dim") if isinstance(cfg, (dict, DictConfig))
                else getattr(cfg, "input_dim", None)
            )
            if self.ambient_dim is None:
                self.ambient_dim = configured_dim
            elif configured_dim is not None and configured_dim != self.ambient_dim:
                raise ValueError("encoder input_dim must match ambient_dim.")
            if self.ambient_dim is None:
                self.ambient_dim = self._infer_ambient_dim()
            if self.ambient_dim is None:
                raise ValueError(
                    "An encoder requires ambient_dim, input_dim in its config, or a datamodule."
                )
            if isinstance(cfg, (dict, DictConfig)):
                cfg["input_dim"] = self.ambient_dim
                encoder = hydra_zen.instantiate(cfg)
            else:
                encoder = cfg
            if not isinstance(encoder, nn.Module) or not isinstance(encoder, HasEncode):
                raise TypeError("encoder must be an nn.Module with encode(x) and latent_dim.")
            if not isinstance(encoder.latent_dim, int) or encoder.latent_dim <= 0:
                raise ValueError("encoder.latent_dim must be a positive integer.")
            if self.latent_dim is not None and self.latent_dim != encoder.latent_dim:
                raise ValueError("latent_dim must match encoder.latent_dim.")
            if self.encoder_pretraining == "gaga" and not isinstance(encoder, GAGANetwork):
                raise TypeError(
                    "GAGA pretraining requires GAGANetwork; "
                    "use encoder_pretraining='none' for a fixed encoder."
                )
            self.latent_dim = encoder.latent_dim
            self.encoder = encoder
            self.preprocessor = Preprocessor(
                mean=torch.zeros(self.ambient_dim), std=torch.ones(self.ambient_dim)
            )
            self._freeze_encoder()

        cfg = self.network_config
        if isinstance(cfg, (dict, DictConfig)):
            flow_dim = self.latent_dim if self.encoder is not None else self.ambient_dim
            if cfg.get("input_dim") is None:
                if flow_dim is None:
                    flow_dim = self._infer_ambient_dim()
                if flow_dim is None:
                    raise ValueError("Flow input_dim requires a config dimension or datamodule.")
                cfg["input_dim"] = flow_dim
            elif flow_dim is not None and cfg["input_dim"] != flow_dim:
                raise ValueError(
                    "Flow input_dim must match the encoder latent_dim "
                    "(or ambient_dim without an encoder)."
                )
            network = hydra_zen.instantiate(cfg)
        else:
            network = cfg
        flow_dim = getattr(network, "input_dim", None)
        if self.encoder is not None and flow_dim is not None and flow_dim != self.latent_dim:
            raise ValueError("Flow input_dim must match encoder.latent_dim.")
        if self.encoder is None:
            self.ambient_dim = self.ambient_dim or flow_dim
            self.latent_dim = self.ambient_dim
        self.network = network
        for name, config in (("network", self.network_config), ("encoder", self.encoder_config)):
            if isinstance(config, (dict, DictConfig)):
                # Resolve while the DictConfig still has its interpolation parent.
                cfg = config if isinstance(config, DictConfig) else OmegaConf.create(config)
                self.hparams[name] = OmegaConf.to_container(cfg, resolve=True)
        self.hparams["ambient_dim"] = self.ambient_dim
        self.hparams["latent_dim"] = self.latent_dim
        self.hparams["encoder_pretraining"] = self.encoder_pretraining
        logger.info("MIOFlow network: %s", self.network.__class__.__name__)

    def _infer_ambient_dim(self):
        if self.datamodule is None:
            return None
        batch = next(iter(self.datamodule.train_dataloader()))
        data = batch["data"] if isinstance(batch, dict) else batch[0]
        return data.shape[1]

    def _freeze_encoder(self):
        if self.encoder is not None:
            self.encoder.requires_grad_(False)
            for parameter in self.encoder.parameters():
                parameter.grad = None
            self.encoder.eval()

    def train(self, mode=True):
        """Keep encoder dropout/BatchNorm fixed when Lightning trains the flow."""
        super().train(mode)
        self._freeze_encoder()
        return self

    @property
    def gaga_network(self):
        """Compatibility alias for callers of the original composed GAGA API."""
        return self.encoder

    @property
    def _gaga_preprocessor(self):
        return self.preprocessor

    def on_fit_start(self):
        """Fit only unfinished encoder stages, after Lightning restores state."""
        if not self._time_span_fitted.item() and self.datamodule is not None:
            times = [
                time
                for batch in self.datamodule.train_dataloader()
                for _, time in self._group_by_time(batch)
            ]
            if times:
                self._time_span.copy_(self._time_span.new_tensor([min(times), max(times)]))
                self._time_span_fitted.fill_(True)
        if not self._encoder_fitted.item():
            self._fit_gaga()
            self._encoder_fitted.fill_(True)
        self._freeze_encoder()

    def _fit_gaga(self) -> None:
        """Fit statistics and GAGA in place; never construct or replace modules."""
        if self.datamodule is None:
            raise ValueError("GAGA pretraining requires a datamodule.")
        all_data = []
        for batch in self.datamodule.train_dataloader():
            data = batch["data"] if isinstance(batch, dict) else batch[0]
            all_data.append(data)
        data = torch.cat(all_data, dim=0).to(self.device)
        gt_distances = self._compute_gaga_target_distances(data.detach().cpu().numpy())
        self.preprocessor.mean.copy_(data.mean(dim=0))
        self.preprocessor.std.copy_(data.std(dim=0).clamp(min=1e-8))
        self.preprocessor.dist_std.fill_(max(float(gt_distances.std()), 1e-8))
        self._train_gaga_two_phase(data, gt_distances)

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
        data = data.to(device)
        x_norm = self.preprocessor.normalize(data)

        triu_idx = np.triu_indices(gt_distances.shape[0], k=1)
        gt_upper = torch.tensor(gt_distances[triu_idx], dtype=x_norm.dtype, device=device)
        gt_upper = self.preprocessor.normalize_dist(gt_upper)

        def _set_requires_grad(module: nn.Module, flag: bool) -> None:
            for p in module.parameters():
                p.requires_grad_(flag)

        self.encoder.train()

        # Phase 1: distance-preserving encoder, decoder frozen.
        _set_requires_grad(self.encoder.encoder, True)
        _set_requires_grad(self.encoder.decoder, False)
        opt1 = torch.optim.Adam(self.encoder.encoder.parameters(), lr=self.gaga_lr)
        for _ in range(self.gaga_encoder_epochs):
            opt1.zero_grad()
            z = self.encoder.encode(x_norm)
            loss = self.gaga_dist_weight_phase1 * gaga_distance_loss(z, gt_upper)
            loss.backward()
            opt1.step()

        # Phase 2: reconstruction decoder, encoder frozen.
        _set_requires_grad(self.encoder.encoder, False)
        _set_requires_grad(self.encoder.decoder, True)
        opt2 = torch.optim.Adam(self.encoder.decoder.parameters(), lr=self.gaga_lr)
        for _ in range(self.gaga_decoder_epochs):
            opt2.zero_grad()
            with torch.no_grad():
                z = self.encoder.encode(x_norm)
            x_hat = self.encoder.decode(z)
            loss = self.gaga_recon_weight_phase2 * gaga_reconstruction_loss(x_hat, x_norm)
            loss.backward()
            opt2.step()

        # Frozen thereafter -- never jointly fine-tuned with the ODE flow.
        _set_requires_grad(self.encoder, False)
        self.encoder.eval()

    def _encode_coordinates(self, x: Tensor) -> Tensor:
        """Map to fixed flow coordinates, or use ambient coordinates without an encoder."""
        if self.encoder is None:
            return x
        return self._encode_with_encoder(x)

    def _encode_with_encoder(self, x: Tensor) -> Tensor:
        assert self.encoder is not None, "Encoder not configured. Call configure_model() first."
        with torch.no_grad():
            x_norm = self.preprocessor.normalize(x)
            return self.encoder.encode(x_norm)

    def _decode_coordinates(self, z: Tensor) -> Tensor:
        assert self.encoder is not None, "Encoder not configured. Call configure_model() first."
        with torch.no_grad():
            x_norm = self.encoder.decode(z)
            return self.preprocessor.unnormalize(x_norm)

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
                labels = batch.get("labels")
            if labels is None:
                labels = batch.get("label")
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

            X_start = self._encode_coordinates(X_start)
            X_end = self._encode_coordinates(X_end)

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

        X_0 = self._encode_coordinates(X_0)

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

            X_true = self._encode_coordinates(X_true)

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
            X_0_latent = self._encode_coordinates(X_0_sample)
            if hasattr(self.network, "reset_momentum"):
                self.network.reset_momentum()
            trajectory = odeint(self.network, X_0_latent, t_bins)
            if self.encoder is not None and self.supports_ambient_trajectories:
                n_bins_, n_traj_, d_latent = trajectory.shape
                flat_ambient = self._decode_coordinates(trajectory.reshape(-1, d_latent))
                trajectory = flat_ambient.reshape(n_bins_, n_traj_, -1)
            self._trajectories = trajectory
        self.network.train()
        logger.info(f"Trajectories generated: shape={self._trajectories.shape}")

    def encode(self, x: Tensor, t_start: float | None = None, t_end: float | None = None) -> Tensor:
        """Integrate x from t_start to t_end, return endpoint positions.

        This produces (n, d) embeddings compatible with the pipeline. When
        an encoder is configured, ``x`` is first encoded and the returned
        endpoint stays in latent space (``d == encoder.latent_dim``).

        Args:
            x: Ambient-space input points.
            t_start: Integration start time. Defaults (with ``t_end``) to the
                minimum training time, persisted in the checkpoint.
            t_end: Integration end time. Defaults to the maximum training time.
                Before fitting, infer from the datamodule or use [0, 1].
        """
        from torchdiffeq import odeint

        assert self.network is not None, "Network not configured. Call setup() first."

        if t_start is not None and t_end is not None:
            t_span = torch.tensor([t_start, t_end], device=x.device, dtype=torch.float32)
        else:
            # Get time range from data if available
            t_span = self._time_span.to(device=x.device)
            if not self._time_span_fitted.item() and self.datamodule is not None:
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
            z = self._encode_coordinates(x)
            if hasattr(self.network, "reset_momentum"):
                self.network.reset_momentum()
            trajectory = odeint(self.network, z, t_span)
        return trajectory[-1]  # endpoint positions

    @property
    def trajectories(self) -> Tensor | None:
        """Paths (n_bins, n_traj, d); ambient if decodable, otherwise latent."""
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
            optimizer = self.optimizer_config(self.network.parameters())
        else:
            optimizer_partial = hydra_zen.instantiate(self.optimizer_config)
            optimizer = optimizer_partial(self.network.parameters())
        return optimizer
