import numpy as np


class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        ## TODO: Implement PCA
        # Perform PCA on the data
        return np.random.rand(data.shape[0], self.n_components)