"""Merging module for multi-channel embeddings.

Merges embeddings from multiple channels using various strategies.
Accepts embeddings from:
1. In-memory dict of tensors
2. DataModule with get_embeddings() method (e.g., PrecomputedDataModule)

Strategies:
    Simple (no projection):
    - concat: Concatenate embeddings (default)
    - weighted_sum: L2-normalize, scale by weights, sum (requires equal dims)
    - mean: Average embeddings (requires equal dims)

    Linear projection:
    - concat_pca: Concat → PCA to target_dim
    - modality_proj: PCA each channel to common dim → concat or mean
    - svd: Concat → truncated SVD to target_dim

Example (in-memory):
    >>> merger = MergingModule(
    ...     embeddings={"dna": dna_tensor, "protein": protein_tensor},
    ...     strategy="concat",
    ... )
    >>> merged = merger.fit_transform(dummy)

Example (projection-based):
    >>> merger = MergingModule(
    ...     embeddings={"dna": dna_tensor, "protein": protein_tensor},
    ...     strategy="concat_pca",
    ...     target_dim=256,
    ... )
    >>> merged = merger.fit_transform(dummy)
    >>> loadings = merger.get_loadings()  # Per-channel contribution

Example (from datamodule):
    >>> dm = PrecomputedDataModule(path="embs/", channels=["dna", "protein"])
    >>> merger = MergingModule(strategy="concat", datamodule=dm)
    >>> merged = merger.fit_transform(dummy)

Example (Hydra CLI):
    python -m manylatents.main \\
        data=precomputed_embeddings \\
        algorithms/latent=merging \\
        algorithms.latent.strategy=concat_pca \\
        algorithms.latent.target_dim=256
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD

from .latent_module_base import LatentModule


@dataclass
class ChannelLoadings:
    """Per-channel contribution to fused components.

    Attributes:
        channel_ranges: Dict mapping channel name to (start, end) indices in concat.
        components: Projection matrix (n_features, target_dim) or None.
        explained_variance_ratio: Variance explained per component, or None.
        channel_contributions: Dict mapping channel → contribution matrix.
            For concat_pca/svd: submatrix of components for that channel's dims.
            For modality_proj: the channel's own PCA components.
    """

    channel_ranges: Dict[str, tuple]
    components: Optional[np.ndarray]
    explained_variance_ratio: Optional[np.ndarray]
    channel_contributions: Dict[str, np.ndarray]


class MergingModule(LatentModule):
    """Merge multi-channel embeddings into a single representation.

    Supports multiple merging strategies:

    Simple (no projection):
    - concat: Concatenate embeddings (default)
    - weighted_sum: L2-normalize, scale by weights, sum (requires equal dims)
    - mean: Average embeddings (requires equal dims)

    Linear projection:
    - concat_pca: Concat → PCA to target_dim. Interpretable loadings show
        which original dimensions (and thus channels) contribute to each component.
    - modality_proj: PCA each channel to common dim → concat or mean.
        Use `proj_aggregation` to control final step ('concat' or 'mean').
    - svd: Concat → truncated SVD to target_dim. Similar to concat_pca but
        centers data differently (mean-centered vs not).

    Embeddings can be provided:
    - Directly via `embeddings` parameter (in-memory)
    - Via `datamodule.get_embeddings()` (file-based)

    Args:
        embeddings: Dict mapping channel names to tensors. If provided,
            used directly instead of datamodule.
        strategy: Merging strategy (see above).
        channels: Which channels to merge. None = all available.
        weights: Channel weights for 'weighted_sum' strategy.
            Missing channels default to 1.0.
        normalize: L2-normalize each channel before merging.
        target_dim: Output dimension for projection strategies.
            Required for concat_pca, modality_proj, svd.
        proj_aggregation: For modality_proj, how to combine projected channels.
            'concat' (default) or 'mean'.
        n_components: Expected output dimension. Auto-computed if None.
        **kwargs: Passed to LatentModule (datamodule, init_seed, etc.)

    Attributes:
        channel_dims: Dict of channel dimensions after setup.
        components_: Projection matrix after fit (for projection strategies).
        explained_variance_ratio_: Variance explained per component.
        channel_projections_: For modality_proj, per-channel PCA objects.
    """

    STRATEGIES = (
        # Simple
        "concat",
        "weighted_sum",
        "mean",
        # Linear projection
        "concat_pca",
        "modality_proj",
        "svd",
    )

    # Strategies that require fitting (projection-based)
    PROJECTION_STRATEGIES = ("concat_pca", "modality_proj", "svd")

    def __init__(
        self,
        embeddings: Optional[Dict[str, Union[Tensor, np.ndarray]]] = None,
        strategy: str = "concat",
        channels: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        normalize: bool = False,
        target_dim: Optional[int] = None,
        proj_aggregation: str = "concat",
        n_components: Optional[int] = None,
        **kwargs,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"strategy must be one of {self.STRATEGIES}, got '{strategy}'"
            )

        # Validate projection strategies require target_dim
        if strategy in self.PROJECTION_STRATEGIES and target_dim is None:
            raise ValueError(
                f"strategy='{strategy}' requires target_dim parameter. "
                f"Example: MergingModule(strategy='{strategy}', target_dim=256)"
            )

        if proj_aggregation not in ("concat", "mean"):
            raise ValueError(
                f"proj_aggregation must be 'concat' or 'mean', got '{proj_aggregation}'"
            )

        super().__init__(n_components=n_components or 0, **kwargs)

        # Convert numpy to torch if needed
        if embeddings is not None:
            self._embeddings = {
                k: torch.from_numpy(v).float() if isinstance(v, np.ndarray) else v.float()
                for k, v in embeddings.items()
            }
        else:
            self._embeddings = None

        self._strategy = strategy
        self._channels = channels
        self._weights = weights or {}
        self._normalize = normalize
        self._target_dim = target_dim
        self._proj_aggregation = proj_aggregation

        # Set after first transform
        self.channel_dims: Dict[str, int] = {}

        # Projection attributes (set after fit for projection strategies)
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.channel_projections_: Dict[str, PCA] = {}
        self._channel_ranges: Dict[str, tuple] = {}
        self._pca: Optional[PCA] = None
        self._svd: Optional[TruncatedSVD] = None
        self._mean: Optional[np.ndarray] = None

    def _get_embeddings(self) -> Dict[str, Tensor]:
        """Get embeddings from in-memory dict or datamodule."""
        if self._embeddings is not None:
            return self._embeddings

        if self.datamodule is None:
            raise ValueError(
                "MergingModule requires either `embeddings` dict or `datamodule` "
                "with get_embeddings() method."
            )

        if not hasattr(self.datamodule, "get_embeddings"):
            raise ValueError(
                f"Datamodule {type(self.datamodule).__name__} has no get_embeddings(). "
                "Use PrecomputedDataModule with channels= parameter, or pass "
                "embeddings= directly."
            )

        return self.datamodule.get_embeddings()

    def fit(self, x: Tensor, y: Tensor | None = None) -> None:
        """Fit projection models for projection-based strategies.

        For simple strategies (concat, weighted_sum, mean), this is a no-op.
        For projection strategies, this fits the projection model(s).

        Args:
            x: Input tensor (ignored - embeddings from __init__ or datamodule)
            y: Optional labels (ignored - MergingModule is unsupervised)
        """
        if self._strategy not in self.PROJECTION_STRATEGIES:
            # Simple strategies don't need fitting
            self._is_fitted = True
            return

        # Get embeddings and prepare for projection fitting
        all_embeddings = self._get_embeddings()
        channels = self._channels or list(all_embeddings.keys())

        # Validate channels exist
        missing = set(channels) - set(all_embeddings.keys())
        if missing:
            raise ValueError(
                f"Channels not found: {missing}. "
                f"Available: {list(all_embeddings.keys())}"
            )

        # Collect embeddings and record dimensions/ranges
        embeddings_list = []
        offset = 0
        for ch in channels:
            emb = all_embeddings[ch]
            if isinstance(emb, Tensor):
                emb = emb.cpu().numpy()
            dim = emb.shape[-1]
            self.channel_dims[ch] = dim
            self._channel_ranges[ch] = (offset, offset + dim)
            offset += dim
            embeddings_list.append(emb)

        # Apply normalization if requested
        if self._normalize:
            embeddings_list = [
                emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)
                for emb in embeddings_list
            ]

        # Fit based on strategy
        if self._strategy == "concat_pca":
            self._fit_concat_pca(embeddings_list, channels)
        elif self._strategy == "modality_proj":
            self._fit_modality_proj(embeddings_list, channels)
        elif self._strategy == "svd":
            self._fit_svd(embeddings_list, channels)

        self._is_fitted = True

    def _fit_concat_pca(
        self, embeddings_list: List[np.ndarray], channels: List[str]
    ) -> None:
        """Fit PCA on concatenated embeddings."""
        concatenated = np.concatenate(embeddings_list, axis=-1)

        self._pca = PCA(n_components=self._target_dim, random_state=self.init_seed)
        self._pca.fit(concatenated)

        self.components_ = self._pca.components_.T  # (n_features, target_dim)
        self.explained_variance_ratio_ = self._pca.explained_variance_ratio_
        self._mean = self._pca.mean_

    def _fit_modality_proj(
        self, embeddings_list: List[np.ndarray], channels: List[str]
    ) -> None:
        """Fit per-channel PCA projections."""
        for ch, emb in zip(channels, embeddings_list):
            # Clamp target_dim to channel dimension
            n_comp = min(self._target_dim, emb.shape[-1])
            pca = PCA(n_components=n_comp, random_state=self.init_seed)
            pca.fit(emb)
            self.channel_projections_[ch] = pca

        # Compute aggregate output dimension
        if self._proj_aggregation == "concat":
            self.n_components = sum(
                min(self._target_dim, self.channel_dims[ch]) for ch in channels
            )
        else:  # mean
            self.n_components = self._target_dim

    def _fit_svd(
        self, embeddings_list: List[np.ndarray], channels: List[str]
    ) -> None:
        """Fit truncated SVD on concatenated embeddings."""
        concatenated = np.concatenate(embeddings_list, axis=-1)

        self._svd = TruncatedSVD(
            n_components=self._target_dim, random_state=self.init_seed
        )
        self._svd.fit(concatenated)

        self.components_ = self._svd.components_.T  # (n_features, target_dim)
        self.explained_variance_ratio_ = self._svd.explained_variance_ratio_

    def transform(self, x: Tensor) -> Tensor:
        """Merge embeddings from all channels.

        Args:
            x: Input tensor (ignored - embeddings from __init__ or datamodule)

        Returns:
            Merged embeddings of shape (N, merged_dim)
        """
        all_embeddings = self._get_embeddings()

        # Select channels
        channels = self._channels or list(all_embeddings.keys())

        # Validate
        missing = set(channels) - set(all_embeddings.keys())
        if missing:
            raise ValueError(
                f"Channels not found: {missing}. "
                f"Available: {list(all_embeddings.keys())}"
            )

        # Collect in order
        embeddings = [all_embeddings[ch] for ch in channels]

        # Record dimensions (if not already done in fit)
        if not self.channel_dims:
            self.channel_dims = {ch: e.shape[-1] for ch, e in zip(channels, embeddings)}

        # Normalize if requested
        if self._normalize:
            embeddings = [
                torch.nn.functional.normalize(e, p=2, dim=-1)
                for e in embeddings
            ]

        # Apply strategy
        if self._strategy == "concat":
            merged = torch.cat(embeddings, dim=-1)

        elif self._strategy == "weighted_sum":
            self._validate_equal_dims(channels, embeddings)
            weights = self._compute_weights(channels)
            merged = sum(w * e for w, e in zip(weights, embeddings))

        elif self._strategy == "mean":
            self._validate_equal_dims(channels, embeddings)
            merged = torch.stack(embeddings, dim=0).mean(dim=0)

        elif self._strategy == "concat_pca":
            merged = self._transform_concat_pca(embeddings)

        elif self._strategy == "modality_proj":
            merged = self._transform_modality_proj(embeddings, channels)

        elif self._strategy == "svd":
            merged = self._transform_svd(embeddings)

        else:
            raise ValueError(f"Unknown strategy: {self._strategy}")

        self.n_components = merged.shape[-1]
        return merged

    def _transform_concat_pca(self, embeddings: List[Tensor]) -> Tensor:
        """Transform using fitted PCA on concatenated embeddings."""
        if self._pca is None:
            raise RuntimeError("Must call fit() before transform() for concat_pca")

        # Convert to numpy and concatenate
        concat_np = np.concatenate(
            [e.cpu().numpy() if isinstance(e, Tensor) else e for e in embeddings],
            axis=-1,
        )

        # Apply PCA transform
        projected = self._pca.transform(concat_np)
        return torch.from_numpy(projected).float()

    def _transform_modality_proj(
        self, embeddings: List[Tensor], channels: List[str]
    ) -> Tensor:
        """Transform each channel with its PCA, then aggregate."""
        if not self.channel_projections_:
            raise RuntimeError("Must call fit() before transform() for modality_proj")

        projected_channels = []
        for ch, emb in zip(channels, embeddings):
            emb_np = emb.cpu().numpy() if isinstance(emb, Tensor) else emb
            pca = self.channel_projections_[ch]
            proj = pca.transform(emb_np)
            projected_channels.append(torch.from_numpy(proj).float())

        if self._proj_aggregation == "concat":
            return torch.cat(projected_channels, dim=-1)
        else:  # mean
            # Pad to common dimension if needed
            max_dim = max(p.shape[-1] for p in projected_channels)
            padded = []
            for p in projected_channels:
                if p.shape[-1] < max_dim:
                    pad = torch.zeros(p.shape[0], max_dim - p.shape[-1])
                    p = torch.cat([p, pad], dim=-1)
                padded.append(p)
            return torch.stack(padded, dim=0).mean(dim=0)

    def _transform_svd(self, embeddings: List[Tensor]) -> Tensor:
        """Transform using fitted SVD on concatenated embeddings."""
        if self._svd is None:
            raise RuntimeError("Must call fit() before transform() for svd")

        # Convert to numpy and concatenate
        concat_np = np.concatenate(
            [e.cpu().numpy() if isinstance(e, Tensor) else e for e in embeddings],
            axis=-1,
        )

        # Apply SVD transform
        projected = self._svd.transform(concat_np)
        return torch.from_numpy(projected).float()

    def _validate_equal_dims(self, channels: List[str], embeddings: List[Tensor]) -> None:
        """Validate all embeddings have equal dimensions."""
        dims = [e.shape[-1] for e in embeddings]
        if len(set(dims)) > 1:
            dim_info = dict(zip(channels, dims))
            raise ValueError(
                f"{self._strategy} requires equal dimensions, got {dim_info}. "
                f"Use strategy='concat' or project to common dimension first."
            )

    def _compute_weights(self, channels: List[str]) -> List[float]:
        """Compute normalized weights for channels."""
        weights = [self._weights.get(ch, 1.0) for ch in channels]
        total = sum(weights)
        return [w / total for w in weights]

    def get_channel_embeddings(self) -> Dict[str, Tensor]:
        """Return individual channel embeddings (before merging)."""
        return self._get_embeddings()

    def get_loadings(self) -> ChannelLoadings:
        """Get interpretability information about channel contributions.

        Returns per-channel contribution to each fused component. For projection
        strategies, this shows which original dimensions (and thus channels)
        contribute most to each output component.

        Returns:
            ChannelLoadings dataclass with:
            - channel_ranges: Dict mapping channel → (start, end) in concat
            - components: Full projection matrix (n_features, target_dim)
            - explained_variance_ratio: Variance per component
            - channel_contributions: Dict of per-channel contribution matrices

        Raises:
            RuntimeError: If called before fit() or for non-projection strategies.

        Example:
            >>> merger = MergingModule(
            ...     embeddings={"dna": dna, "protein": prot},
            ...     strategy="concat_pca", target_dim=128
            ... )
            >>> merger.fit_transform(dummy)
            >>> loadings = merger.get_loadings()
            >>> # See how much DNA contributes to each component
            >>> dna_importance = np.linalg.norm(
            ...     loadings.channel_contributions["dna"], axis=0
            ... )
        """
        if self._strategy not in self.PROJECTION_STRATEGIES:
            raise RuntimeError(
                f"get_loadings() only available for projection strategies "
                f"{self.PROJECTION_STRATEGIES}, not '{self._strategy}'"
            )

        if not self._is_fitted:
            raise RuntimeError("Must call fit() before get_loadings()")

        # Build channel contributions
        channel_contributions = {}

        if self._strategy in ("concat_pca", "svd"):
            # Extract submatrix for each channel from full components
            for ch, (start, end) in self._channel_ranges.items():
                channel_contributions[ch] = self.components_[start:end, :]

        elif self._strategy == "modality_proj":
            # Each channel has its own PCA components
            for ch, pca in self.channel_projections_.items():
                channel_contributions[ch] = pca.components_.T

        return ChannelLoadings(
            channel_ranges=self._channel_ranges.copy(),
            components=self.components_,
            explained_variance_ratio=self.explained_variance_ratio_,
            channel_contributions=channel_contributions,
        )

    def channel_importance(self) -> Dict[str, float]:
        """Compute relative importance of each channel in the fused representation.

        For projection strategies, importance is measured as the Frobenius norm
        of that channel's contribution to the projection matrix, normalized by
        total contribution.

        Returns:
            Dict mapping channel name to importance score (0-1, sums to 1).

        Raises:
            RuntimeError: If called before fit() or for non-projection strategies.

        Example:
            >>> merger = MergingModule(
            ...     embeddings={"dna": dna, "protein": prot},
            ...     strategy="concat_pca", target_dim=128
            ... )
            >>> merger.fit_transform(dummy)
            >>> importance = merger.channel_importance()
            >>> print(importance)  # e.g., {"dna": 0.65, "protein": 0.35}
        """
        loadings = self.get_loadings()

        # Compute Frobenius norm for each channel
        norms = {
            ch: np.linalg.norm(contrib, "fro")
            for ch, contrib in loadings.channel_contributions.items()
        }

        # Normalize to sum to 1
        total = sum(norms.values())
        return {ch: norm / total for ch, norm in norms.items()}

    def __repr__(self) -> str:
        channels_str = self._channels or "all"
        parts = [
            f"strategy='{self._strategy}'",
            f"channels={channels_str}",
            f"normalize={self._normalize}",
        ]
        if self._target_dim is not None:
            parts.append(f"target_dim={self._target_dim}")
        if self._strategy == "modality_proj":
            parts.append(f"proj_aggregation='{self._proj_aggregation}'")
        return f"MergingModule({', '.join(parts)})"
