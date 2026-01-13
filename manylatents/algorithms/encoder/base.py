"""Base class for foundation model encoders.

Foundation encoders are pretrained models that transform input data
(sequences, text, images, etc.) into dense embedding representations.
Unlike LatentModule (which learns representations via fit/transform),
foundation encoders are pretrained and only perform inference.

Examples:
    - Biological: ESM3 (protein), Orthrus (RNA), Evo2 (DNA)
    - Language: LLaMA, BERT, GPT
    - Vision: CLIP, DINOv2
    - Audio: Whisper
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import torch
from torch import Tensor


class FoundationEncoder(ABC):
    """Abstract base class for pretrained foundation model encoders.

    Foundation encoders transform raw input (sequences, text, images, etc.)
    into dense embedding vectors using pretrained models. They do not
    require training - only inference.

    Subclasses must implement:
        - encode(): Single input encoding
        - embedding_dim: Output embedding dimensionality
        - modality: Input modality type

    Subclasses should implement:
        - encode_batch(): Batch encoding (default calls encode() in loop)

    Example:
        >>> encoder = MyEncoder(model_path="/path/to/weights")
        >>> embedding = encoder.encode("MKFGVRA")  # Single sequence
        >>> embeddings = encoder.encode_batch(["MKFGVRA", "MKTAYIA"])  # Batch
        >>> print(encoder.embedding_dim)  # e.g., 1536
        >>> print(encoder.modality)  # e.g., "protein"
    """

    def __init__(self, device: str = "cuda"):
        """Initialize the encoder.

        Args:
            device: Device to run inference on ("cuda" or "cpu").
        """
        self.device = device
        self._model = None

    @abstractmethod
    def encode(self, input: Any) -> Tensor:
        """Encode a single input into embedding space.

        Args:
            input: Raw input data (e.g., sequence string, image tensor).

        Returns:
            Embedding tensor of shape (embedding_dim,) or (1, embedding_dim).
        """
        pass

    def encode_batch(self, inputs: List[Any]) -> Tensor:
        """Encode a batch of inputs into embedding space.

        Default implementation calls encode() for each input.
        Subclasses should override for efficient batched inference.

        Args:
            inputs: List of raw inputs.

        Returns:
            Embedding tensor of shape (batch_size, embedding_dim).
        """
        embeddings = [self.encode(inp) for inp in inputs]
        return torch.stack([e.squeeze(0) for e in embeddings], dim=0)

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of output embeddings."""
        pass

    @property
    @abstractmethod
    def modality(self) -> str:
        """Return the input modality type.

        Standard modalities:
            - "protein": Amino acid sequences
            - "rna": RNA nucleotide sequences
            - "dna": DNA nucleotide sequences
            - "text": Natural language text
            - "image": Image data
            - "audio": Audio data
        """
        pass

    @property
    def model(self) -> Any:
        """Return the underlying model (lazy loaded)."""
        return self._model

    def to(self, device: str) -> "FoundationEncoder":
        """Move encoder to specified device.

        Args:
            device: Target device ("cuda" or "cpu").

        Returns:
            Self for method chaining.
        """
        self.device = device
        if self._model is not None and hasattr(self._model, "to"):
            self._model = self._model.to(device)
        return self

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"modality={self.modality!r}, "
            f"embedding_dim={self.embedding_dim}, "
            f"device={self.device!r})"
        )
