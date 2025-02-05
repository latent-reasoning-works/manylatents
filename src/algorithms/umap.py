import numpy as np
import umap

class UMAP:
    def __init__(self, 
                 n_components: int = 2, 
                 n_neighbors: int = 15, 
                 random_state: int = None, 
                 kwargs: dict = None):
        """
        UMAP Wrapper for dimensionality reduction.

        Args:
            n_components (int): Number of dimensions to reduce to.
            n_neighbors (int): Number of neighbors considered for manifold approximation.
            random_state (int, optional): Random seed for reproducibility.
            kwargs (dict): Additional UMAP parameters (e.g., min_dist, metric, spread, learning_rate, etc.).
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.umap_obj = None 
        self.umap_params = kwargs if kwargs else {} 

    def _create_umap(self):
        """Instantiate the UMAP object with stored parameters."""
        return umap.UMAP(
                            n_components=self.n_components, 
                            n_neighbors=self.n_neighbors, 
                            random_state=self.random_state, 
                            **self.umap_params
                        )

    def fit(self, data: np.ndarray):
        """Fit UMAP on the input data."""
        self.umap_obj = self._create_umap()
        self.umap_obj.fit(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using the fitted UMAP model."""
        if self.umap_obj is None:
            raise ValueError("UMAP model is not fitted yet. Call `fit()` first.")
        return self.umap_obj.transform(data)

    def fit_transform(self, data: np.ndarray, fit_idx: np.ndarray = None, transform_idx: np.ndarray = None) -> np.ndarray:
        """Fit UMAP and transform the data. If indices are provided, fit on `fit_idx` and transform `transform_idx`."""
        if fit_idx is None or transform_idx is None:
            self.umap_obj = self._create_umap()
            return self.umap_obj.fit_transform(data)

        # Fit UMAP on the selected indices
        self.fit(data[fit_idx])

        # Transform both fit and transform indices
        umap_fit_emb = self.transform(data[fit_idx])
        umap_transform_emb = self.transform(data[transform_idx])

        # Concatenate transformed embeddings into original order
        umap_emb = np.zeros((data.shape[0], umap_fit_emb.shape[1]), dtype=np.float32)
        umap_emb[fit_idx] = umap_fit_emb
        umap_emb[transform_idx] = umap_transform_emb

        return umap_emb
