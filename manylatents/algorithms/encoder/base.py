"""Base class for foundation model encoders.

Foundation encoders are pretrained models that transform input data
(sequences, text, images, etc.) into dense embedding representations.
They inherit from LatentModule to integrate with manylatents experiment
pipeline via fit/transform interface.

Examples:
    - Biological: ESM3 (protein), Orthrus (RNA), Evo2 (DNA)
    - Language: LLaMA, BERT, GPT
    - Vision: CLIP, DINOv2
    - Audio: Whisper

Cluster Weight Loading:
    Most foundation models support local weight loading for HPC clusters
    where internet access is restricted. Each encoder supports weights_path
    or equivalent parameter.

    Mila cluster examples:
        - ESM3: /network/weights/esm3-sm-open-v1/esm3-sm-open-v1/
        - Orthrus: /network/weights/orthrus/Orthrus/models/
        - Evo2: /network/weights/savanna-evo2-1b-base/

    Environment variables:
        - ESM3: Set INFRA_PROVIDER=local when running from weights dir
"""

from abc import abstractmethod
from typing import Any, List, Optional

import torch
from torch import Tensor

from manylatents.algorithms.latent.latent_module_base import LatentModule


class FoundationEncoder(LatentModule):
    """Abstract base class for pretrained foundation model encoders.

    Foundation encoders transform raw input (sequences, text, images, etc.)
    into dense embedding vectors using pretrained models. They inherit from
    LatentModule to work with manylatents experiment pipeline:

        - fit(): No-op (pretrained, nothing to fit)
        - transform(): Gets sequences from datamodule, calls encode_batch()

    Subclasses must implement:
        - _load_model(): Load the pretrained model
        - encode(): Single input encoding
        - modality: Input modality type

    Subclasses should set:
        - self._embedding_dim in __init__

    Example:
        >>> encoder = ESM3Encoder(datamodule=seq_datamodule)
        >>> encoder.fit(dummy_tensor)  # no-op
        >>> embeddings = encoder.transform(dummy_tensor)  # encodes sequences
    """

    def __init__(self, device: str = "cuda", **kwargs):
        """Initialize the encoder.

        Args:
            device: Device to run inference on ("cuda" or "cpu").
            **kwargs: Passed to LatentModule (datamodule, n_components, etc.)
        """
        if 'n_components' not in kwargs:
            kwargs['n_components'] = 0
        super().__init__(**kwargs)
        self.device = device
        self._model = None
        self._embedding_dim: int = 0  # Subclasses set this in __init__

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded. Calls _load_model() if needed."""
        if self._model is None:
            self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """Load the pretrained model. Called lazily on first encode().

        Subclasses should:
        1. Import required packages (with ImportError handling)
        2. Load model from weights_path or HuggingFace
        3. Move model to self.device and set to eval mode
        4. Store model in self._model
        """
        pass

    @abstractmethod
    def encode(self, input: Any) -> Tensor:
        """Encode a single input into embedding space.

        Args:
            input: Raw input data (e.g., sequence string, image tensor).

        Returns:
            Embedding tensor of shape (embedding_dim,) or (1, embedding_dim).
        """
        pass

    def encode_batch(
        self,
        inputs: List[Any],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> Tensor:
        """Batched encoding with micro-batch chunking.

        If the subclass implements _tokenize_batch() and _extract_embeddings(),
        uses true GPU batching (single forward pass per micro-batch). Otherwise
        falls back to looping encode() with consolidated CPU transfer.

        Args:
            inputs: List of raw inputs (sequences, etc.)
            batch_size: Micro-batch size for GPU forward passes.
            show_progress: Show tqdm progress bar.

        Returns:
            (N, embedding_dim) tensor on CPU.
        """
        self._ensure_loaded()

        if not self._supports_batched_forward():
            # Fallback: loop with consolidated CPU transfer
            embeddings = []
            iterator = range(0, len(inputs), batch_size)
            if show_progress:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc=f"{self.__class__.__name__} encode")
            for start in iterator:
                chunk = inputs[start:start + batch_size]
                chunk_embs = [self.encode(inp) for inp in chunk]
                embeddings.append(torch.stack(
                    [e.squeeze(0) for e in chunk_embs]
                ).cpu())
            return torch.cat(embeddings, dim=0)

        # True batched inference path with OOM retry
        import logging
        _logger = logging.getLogger(__name__)

        all_embeddings = []
        start = 0
        current_bs = batch_size
        n_total = len(inputs)

        if show_progress:
            from tqdm import tqdm
            pbar = tqdm(total=n_total, desc=f"{self.__class__.__name__} batched")
        else:
            pbar = None

        while start < n_total:
            chunk = inputs[start:start + current_bs]
            try:
                batch_inputs = self._tokenize_batch(chunk)
                with torch.no_grad():
                    embeddings = self._extract_embeddings(batch_inputs)
                all_embeddings.append(embeddings.float().cpu())
                if pbar:
                    pbar.update(len(chunk))
                start += current_bs
                # Restore batch size after successful forward
                current_bs = batch_size
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if current_bs <= 1:
                    raise RuntimeError(
                        f"OOM even at batch_size=1 (seq_len={len(chunk[0]) if chunk else '?'}). "
                        "Reduce max_length or use a smaller model."
                    )
                current_bs = max(1, current_bs // 2)
                _logger.warning(f"OOM at batch_size={current_bs * 2}, retrying with {current_bs}")

        if pbar:
            pbar.close()

        return torch.cat(all_embeddings, dim=0)

    def _supports_batched_forward(self) -> bool:
        """Override to return True if _tokenize_batch/_extract_embeddings are implemented."""
        return False

    def _tokenize_batch(self, inputs: List[Any]) -> dict:
        """Tokenize + pad + stack a batch. Override in subclass."""
        raise NotImplementedError

    def _extract_embeddings(self, batch: dict) -> Tensor:
        """Single forward pass -> pooled embeddings. Override in subclass."""
        raise NotImplementedError

    # --- LatentModule interface ---

    def fit(self, x: Tensor) -> None:
        """No-op fit for pretrained encoders.

        Foundation encoders are pretrained and don't require fitting.
        The input tensor is ignored - sequences come from datamodule.
        """
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """Transform by encoding sequences from datamodule.

        Args:
            x: Input tensor (ignored - sequences come from datamodule).

        Returns:
            Embedding tensor of shape (batch_size, embedding_dim).
        """
        if self.datamodule is None:
            raise ValueError(
                "FoundationEncoder requires datamodule with get_sequences(). "
                "Pass datamodule= when instantiating the encoder."
            )

        if not hasattr(self.datamodule, 'get_sequences'):
            raise ValueError(
                f"Datamodule {type(self.datamodule).__name__} has no get_sequences() method. "
                "Use SequenceDataModule for foundation encoder data."
            )

        sequences = self.datamodule.get_sequences()
        return self.encode_batch(sequences)

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of output embeddings."""
        return self._embedding_dim

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
