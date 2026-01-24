"""Merging module for multi-channel embeddings.

Merges embeddings from multiple channels using various strategies.
Accepts embeddings from:
1. In-memory dict of tensors
2. DataModule with get_embeddings() method (e.g., PrecomputedDataModule)

Example (in-memory):
    >>> merger = MergingModule(
    ...     embeddings={"dna": dna_tensor, "protein": protein_tensor},
    ...     strategy="concat",
    ... )
    >>> merged = merger.fit_transform(dummy)

Example (from datamodule):
    >>> dm = PrecomputedDataModule(path="embs/", channels=["dna", "protein"])
    >>> merger = MergingModule(strategy="concat", datamodule=dm)
    >>> merged = merger.fit_transform(dummy)

Example (Hydra CLI):
    python -m manylatents.main \\
        data=precomputed_embeddings \\
        algorithms/latent=merging \\
        algorithms.latent.strategy=concat
"""
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor
import numpy as np

from .latent_module_base import LatentModule


class MergingModule(LatentModule):
    """Merge multi-channel embeddings into a single representation.

    Supports multiple merging strategies:
    - concat: Concatenate embeddings (default)
    - weighted_sum: L2-normalize, scale by weights, sum
    - mean: Average embeddings (requires equal dimensions)

    Embeddings can be provided:
    - Directly via `embeddings` parameter (in-memory)
    - Via `datamodule.get_embeddings()` (file-based)

    Args:
        embeddings: Dict mapping channel names to tensors. If provided,
            used directly instead of datamodule.
        strategy: Merging strategy ('concat', 'weighted_sum', 'mean')
        channels: Which channels to merge. None = all available.
        weights: Channel weights for 'weighted_sum' strategy.
            Missing channels default to 1.0.
        normalize: L2-normalize each channel before merging.
        n_components: Expected output dimension. Auto-computed if None.
        **kwargs: Passed to LatentModule (datamodule, init_seed, etc.)

    Attributes:
        channel_dims: Dict of channel dimensions after setup.
    """

    STRATEGIES = ("concat", "weighted_sum", "mean")

    def __init__(
        self,
        embeddings: Optional[Dict[str, Union[Tensor, np.ndarray]]] = None,
        strategy: str = "concat",
        channels: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        normalize: bool = False,
        n_components: Optional[int] = None,
        **kwargs,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"strategy must be one of {self.STRATEGIES}, got '{strategy}'"
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

        # Set after first transform
        self.channel_dims: Dict[str, int] = {}

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

    def fit(self, x: Tensor) -> None:
        """No-op fit - merging doesn't require training."""
        self._is_fitted = True

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

        # Record dimensions
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

        else:
            raise ValueError(f"Unknown strategy: {self._strategy}")

        self.n_components = merged.shape[-1]
        return merged

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

    def __repr__(self) -> str:
        channels_str = self._channels or "all"
        return (
            f"MergingModule(strategy='{self._strategy}', "
            f"channels={channels_str}, normalize={self._normalize})"
        )
