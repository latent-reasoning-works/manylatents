import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

class PrecomputedMixin:
    """
    Mixin to handle logic for loading precomputed embeddings alongside raw data.
    """
    def load_precomputed(self, precomputed_path: str, mmap_mode: str = None):
        if precomputed_path:
            abs_path = os.path.abspath(precomputed_path)
            logger.info(f"Resolved precomputed path: {abs_path}")
            if os.path.exists(abs_path):
                logger.info(f"Loading precomputed embeddings from {abs_path}")
                if abs_path.endswith(".npy"):
                    return np.load(abs_path, mmap_mode=mmap_mode)
                elif abs_path.endswith(".csv"):
                    # Use genfromtxt to skip the header and select numeric columns.
                    # Adjust `usecols` as needed; here we assume columns 0 and 1 are numeric.
                    return np.genfromtxt(abs_path, delimiter=",", skip_header=1)
                else:
                    raise ValueError(f"Unsupported file format: {abs_path}")
        logger.info("No precomputed embeddings found or path does not exist.")
        return None

    def get_data(self, idx: int):
        """
        Returns both raw data and precomputed embeddings (if available) for a given index.
        
        Override this method if you need a different return structure.
        """
        # Assume self.original_data is already loaded by the dataset.
        sample_raw = self.original_data[idx]
        sample_precomputed = None
        if hasattr(self, "precomputed_embeddings") and self.precomputed_embeddings is not None:
            sample_precomputed = self.precomputed_embeddings[idx]
        return sample_raw, sample_precomputed