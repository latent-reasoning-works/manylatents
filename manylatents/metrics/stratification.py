import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from typing import Optional
from manylatents.algorithms.latent_module_base import LatentModule
logger = logging.getLogger(__name__)

def kmeans_stratification(embeddings: np.ndarray, 
                        dataset, 
                        module: Optional[LatentModule] = None,
                        random_state=42):  
    """
    Computes the Adjusted Rand Index (ARI) between the KMeans clustering labels and the provided dataset labels.
    """

    if hasattr(dataset, 'metadata') and 'Population' in dataset.metadata:
        n_clusters = len(np.unique(dataset.metadata['Population']))
        train_labels = dataset.metadata['Population']
    else:
        n_clusters = len(np.unique(dataset.metadata))
        train_labels = dataset.metadata

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans_label = kmeans.fit_predict(embeddings)
    ari = adjusted_rand_score(train_labels, kmeans_label)
    return ari
