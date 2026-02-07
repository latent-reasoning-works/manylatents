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
    - concat_pca: Concat -> PCA to target_dim
    - modality_proj: PCA each channel to common dim -> concat or mean
    - svd: Concat -> truncated SVD to target_dim

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
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.decomposition import PCA, TruncatedSVD
from torch import Tensor

from .latent_module_base import LatentModule


@dataclass
class ChannelLoadings:
    """Per-channel contribution to fused components.

    Attributes:
        channel_ranges: Dict mapping channel name to (start, end) indices in concat.
        components: Projection matrix (n_features, target_dim) or None.
        explained_variance_ratio: Variance explained per component, or None.
        channel_contributions: Dict mapping channel -> contribution matrix.
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
    - concat_pca: Concat -> PCA to target_dim. Interpretable loadings show
        which original dimensions (and thus channels) contribute to each component.
    - modality_proj: PCA each channel to common dim -> concat or mean.
        Use `proj_aggregation` to control final step ('concat' or 'mean').
    - svd: Concat -> truncated SVD to target_dim. Similar to concat_pca but
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
        self._validate_init_params(strategy, target_dim, proj_aggregation)
        super().__init__(n_components=n_components or 0, **kwargs)

        self._embeddings = self._convert_embeddings(embeddings)
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
        self._projection_model: Optional[Union[PCA, TruncatedSVD]] = None

    def _validate_init_params(
        self, strategy: str, target_dim: Optional[int], proj_aggregation: str
    ) -> None:
        """Validate constructor parameters."""
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"strategy must be one of {self.STRATEGIES}, got '{strategy}'"
            )
        if strategy in self.PROJECTION_STRATEGIES and target_dim is None:
            raise ValueError(
                f"strategy='{strategy}' requires target_dim parameter. "
                f"Example: MergingModule(strategy='{strategy}', target_dim=256)"
            )
        if proj_aggregation not in ("concat", "mean"):
            raise ValueError(
                f"proj_aggregation must be 'concat' or 'mean', got '{proj_aggregation}'"
            )

    @staticmethod
    def _convert_embeddings(
        embeddings: Optional[Dict[str, Union[Tensor, np.ndarray]]]
    ) -> Optional[Dict[str, Tensor]]:
        """Convert numpy arrays to torch tensors."""
        if embeddings is None:
            return None
        return {
            k: torch.from_numpy(v).float() if isinstance(v, np.ndarray) else v.float()
            for k, v in embeddings.items()
        }

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

    def _prepare_channels(
        self, all_embeddings: Dict[str, Tensor]
    ) -> Tuple[List[str], List[Tensor]]:
        """Select and validate channels, returning ordered lists."""
        channels = self._channels or list(all_embeddings.keys())
        missing = set(channels) - set(all_embeddings.keys())
        if missing:
            raise ValueError(
                f"Channels not found: {missing}. "
                f"Available: {list(all_embeddings.keys())}"
            )
        embeddings = [all_embeddings[ch] for ch in channels]
        return channels, embeddings

    def _record_channel_dims(
        self, channels: List[str], embeddings: List[Union[Tensor, np.ndarray]]
    ) -> None:
        """Record channel dimensions and ranges for later use."""
        offset = 0
        for ch, emb in zip(channels, embeddings):
            dim = emb.shape[-1]
            self.channel_dims[ch] = dim
            self._channel_ranges[ch] = (offset, offset + dim)
            offset += dim

    @staticmethod
    def _normalize_embeddings_np(embeddings: List[np.ndarray]) -> List[np.ndarray]:
        """L2-normalize numpy embeddings."""
        return [
            emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)
            for emb in embeddings
        ]

    @staticmethod
    def _normalize_embeddings_torch(embeddings: List[Tensor]) -> List[Tensor]:
        """L2-normalize torch embeddings."""
        return [torch.nn.functional.normalize(e, p=2, dim=-1) for e in embeddings]

    @staticmethod
    def _to_numpy(embeddings: List[Tensor]) -> List[np.ndarray]:
        """Convert list of tensors to numpy arrays."""
        return [
            e.cpu().numpy() if isinstance(e, Tensor) else e for e in embeddings
        ]

    def fit(self, x: Tensor, y: Tensor | None = None) -> None:
        """Fit projection models for projection-based strategies.

        For simple strategies (concat, weighted_sum, mean), this is a no-op.
        For projection strategies, this fits the projection model(s).

        Args:
            x: Input tensor (ignored - embeddings from __init__ or datamodule)
            y: Optional labels (ignored - MergingModule is unsupervised)
        """
        if self._strategy not in self.PROJECTION_STRATEGIES:
            self._is_fitted = True
            return

        all_embeddings = self._get_embeddings()
        channels, embeddings = self._prepare_channels(all_embeddings)
        embeddings_np = self._to_numpy(embeddings)
        self._record_channel_dims(channels, embeddings_np)

        if self._normalize:
            embeddings_np = self._normalize_embeddings_np(embeddings_np)

        if self._strategy == "modality_proj":
            self._fit_modality_proj(embeddings_np, channels)
        else:
            # concat_pca and svd share the same fitting pattern
            self._fit_concat_projection(embeddings_np, channels)

        self._is_fitted = True

    def _fit_concat_projection(
        self, embeddings_list: List[np.ndarray], channels: List[str]
    ) -> None:
        """Fit PCA or SVD on concatenated embeddings."""
        concatenated = np.concatenate(embeddings_list, axis=-1)

        if self._strategy == "concat_pca":
            model = PCA(n_components=self._target_dim, random_state=self.init_seed)
        else:  # svd
            model = TruncatedSVD(
                n_components=self._target_dim, random_state=self.init_seed
            )

        model.fit(concatenated)
        self._projection_model = model
        self.components_ = model.components_.T  # (n_features, target_dim)
        self.explained_variance_ratio_ = model.explained_variance_ratio_

    def _fit_modality_proj(
        self, embeddings_list: List[np.ndarray], channels: List[str]
    ) -> None:
        """Fit per-channel PCA projections."""
        for ch, emb in zip(channels, embeddings_list):
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

    def transform(self, x: Tensor) -> Tensor:
        """Merge embeddings from all channels.

        Args:
            x: Input tensor (ignored - embeddings from __init__ or datamodule)

        Returns:
            Merged embeddings of shape (N, merged_dim)
        """
        all_embeddings = self._get_embeddings()
        channels, embeddings = self._prepare_channels(all_embeddings)

        if not self.channel_dims:
            self.channel_dims = {ch: e.shape[-1] for ch, e in zip(channels, embeddings)}

        if self._normalize:
            embeddings = self._normalize_embeddings_torch(embeddings)

        merged = self._apply_strategy(channels, embeddings)
        self.n_components = merged.shape[-1]
        return merged

    def _apply_strategy(self, channels: List[str], embeddings: List[Tensor]) -> Tensor:
        """Apply the configured merging strategy."""
        if self._strategy == "concat":
            return torch.cat(embeddings, dim=-1)

        if self._strategy == "weighted_sum":
            self._validate_equal_dims(channels, embeddings)
            weights = self._compute_weights(channels)
            return sum(w * e for w, e in zip(weights, embeddings))

        if self._strategy == "mean":
            self._validate_equal_dims(channels, embeddings)
            return torch.stack(embeddings, dim=0).mean(dim=0)

        if self._strategy == "modality_proj":
            return self._transform_modality_proj(embeddings, channels)

        # concat_pca and svd use the same transform logic
        return self._transform_concat_projection(embeddings)

    def _transform_concat_projection(self, embeddings: List[Tensor]) -> Tensor:
        """Transform using fitted PCA or SVD on concatenated embeddings."""
        if self._projection_model is None:
            strategy_name = self._strategy
            raise RuntimeError(f"Must call fit() before transform() for {strategy_name}")

        concat_np = np.concatenate(self._to_numpy(embeddings), axis=-1)
        projected = self._projection_model.transform(concat_np)
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

        # mean aggregation - pad to common dimension if needed
        max_dim = max(p.shape[-1] for p in projected_channels)
        padded = []
        for p in projected_channels:
            if p.shape[-1] < max_dim:
                pad = torch.zeros(p.shape[0], max_dim - p.shape[-1])
                p = torch.cat([p, pad], dim=-1)
            padded.append(p)
        return torch.stack(padded, dim=0).mean(dim=0)

    def _validate_equal_dims(
        self, channels: List[str], embeddings: List[Tensor]
    ) -> None:
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
            - channel_ranges: Dict mapping channel -> (start, end) in concat
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

        channel_contributions = self._compute_channel_contributions()

        return ChannelLoadings(
            channel_ranges=self._channel_ranges.copy(),
            components=self.components_,
            explained_variance_ratio=self.explained_variance_ratio_,
            channel_contributions=channel_contributions,
        )

    def _compute_channel_contributions(self) -> Dict[str, np.ndarray]:
        """Compute per-channel contribution matrices for loadings."""
        if self._strategy in ("concat_pca", "svd"):
            return {
                ch: self.components_[start:end, :]
                for ch, (start, end) in self._channel_ranges.items()
            }
        # modality_proj
        return {ch: pca.components_.T for ch, pca in self.channel_projections_.items()}

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

        norms = {
            ch: np.linalg.norm(contrib, "fro")
            for ch, contrib in loadings.channel_contributions.items()
        }

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


class DiffusionMerging:
    """Merge multiple diffusion operators into a single target operator.

    Strategies:
    - weighted_interpolation: P* = Σ w_i P_i, then normalize to row-stochastic
    - frobenius_mean: P* = (1/N) Σ P_i (arithmetic mean, closed-form Frobenius)
    - ot_barycenter: Wasserstein barycenter (requires POT library)

    Attributes:
        strategy: Merging strategy
        weights: Optional per-operator weights (normalized internally)
        normalize_output: Whether to ensure output is row-stochastic
    """

    STRATEGIES = ("weighted_interpolation", "frobenius_mean", "ot_barycenter")

    def __init__(
        self,
        strategy: str = "weighted_interpolation",
        weights: Optional[Dict[str, float]] = None,
        normalize_output: bool = True,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")

        self.strategy = strategy
        self.weights = weights or {}
        self.normalize_output = normalize_output

    def merge(self, operators: Dict[str, np.ndarray]) -> np.ndarray:
        """Merge multiple diffusion operators.

        Args:
            operators: Dict mapping operator name to (N, N) array

        Returns:
            Merged operator of shape (N, N)
        """
        if len(operators) == 0:
            raise ValueError("operators dict is empty")

        # Validate all same shape
        shapes = {k: v.shape for k, v in operators.items()}
        unique_shapes = set(shapes.values())
        if len(unique_shapes) > 1:
            raise ValueError(f"All operators must have same shape, got {shapes}")

        if self.strategy == "weighted_interpolation":
            return self._weighted_interpolation(operators)
        elif self.strategy == "frobenius_mean":
            return self._frobenius_mean(operators)
        elif self.strategy == "ot_barycenter":
            return self._ot_barycenter(operators)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _get_weights(self, operators: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Get normalized weights for operators."""
        weights = {k: self.weights.get(k, 1.0) for k in operators}
        total = sum(weights.values())
        return {k: w / total for k, w in weights.items()}

    def _weighted_interpolation(
        self, operators: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Weighted sum of operators, normalized to row-stochastic."""
        weights = self._get_weights(operators)

        merged = sum(w * operators[k] for k, w in weights.items())

        if self.normalize_output:
            row_sums = merged.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-10)
            merged = merged / row_sums

        return merged

    def _frobenius_mean(self, operators: Dict[str, np.ndarray]) -> np.ndarray:
        """Arithmetic mean (closed-form Frobenius barycenter)."""
        ops = list(operators.values())
        merged = np.mean(ops, axis=0)

        if self.normalize_output:
            row_sums = merged.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-10)
            merged = merged / row_sums

        return merged

    def _ot_barycenter(self, operators: Dict[str, np.ndarray]) -> np.ndarray:
        """Wasserstein barycenter using POT library."""
        try:
            import ot
        except ImportError:
            raise ImportError(
                "ot_barycenter strategy requires POT library. "
                "Install with: pip install POT"
            )

        weights = self._get_weights(operators)
        ops = list(operators.values())
        weight_array = np.array([weights[k] for k in operators])

        # Stack operators for POT
        # POT expects distributions as columns
        n = ops[0].shape[0]

        # Use Sinkhorn barycenter on rows
        # Each row of the diffusion operator is a distribution
        merged_rows = []
        for i in range(n):
            # Get i-th row from each operator
            distributions = np.array([op[i, :] for op in ops]).T  # (n, num_ops)

            # Compute barycenter of these distributions
            M = ot.dist(np.arange(n).reshape(-1, 1).astype(float))  # Cost matrix
            M = M / M.max()

            barycenter = ot.bregman.barycenter(
                distributions,
                M,
                reg=0.01,  # Regularization
                weights=weight_array,
            )
            merged_rows.append(barycenter)

        merged = np.array(merged_rows)

        if self.normalize_output:
            row_sums = merged.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-10)
            merged = merged / row_sums

        return merged
