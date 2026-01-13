"""Foundation model encoders for manylatents.

This module provides the base class for pretrained foundation model encoders.
Specific implementations (ESM3, Orthrus, Evo2, LLaMA, etc.) are provided
by extension packages:

    - manylatents-dogma: Biological encoders (DNA/RNA/Protein)
    - manylatents-nlp: Language model encoders (future)
    - manylatents-vision: Vision encoders (future)

Example:
    >>> from manylatents.algorithms.encoder import FoundationEncoder
    >>> from manylatents.dogma.encoders import ESM3Encoder  # From extension
    >>>
    >>> encoder = ESM3Encoder()
    >>> embedding = encoder.encode("MKFGVRA")
"""

from .base import FoundationEncoder

__all__ = ["FoundationEncoder"]
