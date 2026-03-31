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
"""

from abc import abstractmethod
from typing import Any, Dict, List, Union

import torch
from torch import Tensor

from .latent_module_base import LatentModule


class _EmbeddingAccumulator:
    """Accumulates embeddings across micro-batches, handling dict or tensor output."""

    def __init__(self):
        self._store = None
        self._is_dict = None

    def append(self, batch_result):
        if self._is_dict is None:
            self._is_dict = isinstance(batch_result, dict)
            self._store = {k: [] for k in batch_result} if self._is_dict else []
        if self._is_dict:
            for k in self._store:
                self._store[k].append(batch_result[k].float().cpu())
        else:
            self._store.append(batch_result.float().cpu())

    def append_single(self, single_result):
        """Append a single encode() result (squeeze + cpu)."""
        if self._is_dict is None:
            self._is_dict = isinstance(single_result, dict)
            self._store = {k: [] for k in single_result} if self._is_dict else []
        if self._is_dict:
            for k in self._store:
                self._store[k].append(single_result[k].squeeze(0).cpu())
        else:
            self._store.append(single_result.squeeze(0).cpu())

    def result(self) -> Union[Tensor, Dict[str, Tensor]]:
        if self._is_dict:
            return {k: self._combine(v) for k, v in self._store.items()}
        return self._combine(self._store)

    @staticmethod
    def _combine(tensors: list) -> Tensor:
        """Stack 0D/1D tensors, cat 2D+ tensors."""
        if not tensors:
            return torch.tensor([])
        if tensors[0].dim() <= 1:
            return torch.stack(tensors)
        return torch.cat(tensors, dim=0)


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
        if 'n_components' not in kwargs:
            kwargs['n_components'] = 0
        super().__init__(**kwargs)
        self.device = device
        self._model = None
        self._embedding_dim: int = 0

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
    ):
        """Batched encoding with micro-batch chunking.

        If the subclass implements _tokenize_batch() and _extract_embeddings(),
        uses true GPU batching (single forward pass per micro-batch). Otherwise
        falls back to looping encode() with consolidated CPU transfer.

        Args:
            inputs: List of raw inputs (sequences, etc.)
            batch_size: Micro-batch size for GPU forward passes.
            show_progress: Show tqdm progress bar.

        Returns:
            (N, embedding_dim) tensor on CPU, or dict of such tensors for
            multi-layer encoders.
        """
        self._ensure_loaded()
        acc = _EmbeddingAccumulator()

        if not self._supports_batched_forward():
            iterator = range(0, len(inputs), batch_size)
            if show_progress:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc=f"{self.__class__.__name__} encode")
            for start in iterator:
                for inp in inputs[start:start + batch_size]:
                    acc.append_single(self.encode(inp))
            return acc.result()

        import logging
        _logger = logging.getLogger(__name__)

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

                acc.append(embeddings)

                if pbar:
                    pbar.update(len(chunk))
                start += current_bs
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

        return acc.result()

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

    def fit(self, x: Tensor, y: Tensor = None) -> None:
        """No-op fit for pretrained encoders."""
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """Transform by encoding sequences from datamodule."""
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
        """Move encoder to specified device."""
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
