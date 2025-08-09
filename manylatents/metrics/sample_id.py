import logging
import numpy as np
from typing import Optional
from manylatents.algorithms.latent_module_base import LatentModule
logger = logging.getLogger(__name__)

def sample_id(embeddings: np.ndarray, 
              dataset, 
              module: Optional[LatentModule] = None,
              random_state=42):  
    """
    Fetches sample IDs (for downstream analysis)
    """

    if hasattr(dataset, 'metadata') and 'sample_id' in dataset.metadata:
        sample_ids = dataset.metadata['sample_id']
        return sample_ids
    else:
        return None
