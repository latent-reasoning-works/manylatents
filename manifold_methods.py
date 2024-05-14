import os
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import phate
from typing import Tuple, Any

def save_object(obj: Any, filename: str) -> None:
    """Save a Python object to disk using pickle."""
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def load_object(filename: str) -> Any:
    """Load a Python object from disk using pickle."""
    with open(filename, 'rb') as file:
        return pickle.load(file)

def perform_pca(data: np.ndarray, model_filename: str, **kwargs) -> Tuple[PCA, np.ndarray]:
    """Perform PCA or load a pre-fitted PCA model."""
    if os.path.exists(model_filename):
        pca = load_object(model_filename)
    else:
        pca = PCA(**kwargs)
        pca.fit(data)
        save_object(pca, model_filename)
    transformed_data = pca.transform(data)
    return pca, transformed_data

def perform_tsne(data: np.ndarray, model_filename: str, **kwargs) -> Tuple[TSNE, np.ndarray]:
    """Perform TSNE or load a pre-fitted TSNE model."""
    if os.path.exists(model_filename):
        tsne = load_object(model_filename)
    else:
        tsne = TSNE(**kwargs)
        # Note: TSNE does not support fit_transform on a pre-fitted model as it does not have a separate fit method
        transformed_data = tsne.fit_transform(data)
        save_object(tsne, model_filename)
        return tsne, transformed_data
    #transformed_data = tsne.fit_transform(data)  # TSNE must re-run every time as it's stateful
    transformed_data = tsne.embedding_ # assume fitting on the same data!
    return tsne, transformed_data

def perform_phate(data: np.ndarray, model_filename: str, **kwargs) -> Tuple[phate.PHATE, np.ndarray]:
    """Perform PHATE or load a pre-fitted PHATE model."""
    if os.path.exists(model_filename):
        phate_operator = load_object(model_filename)
    else:
        phate_operator = phate.PHATE(**kwargs)
        phate_operator.fit(data) # uses knn and decay
        save_object(phate_operator, model_filename)

    kwargs.pop('knn', None)
    kwargs.pop('decay', None)
    phate_operator.set_params(**kwargs) # add in kwargs
    transformed_data = phate_operator.transform(data)
    return phate_operator, transformed_data
