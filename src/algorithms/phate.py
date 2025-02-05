import numpy as np
import phate

class PHATE:
    def __init__(self, n_components: int = 2, knn: int = 5, t: str = 'auto', gamma: float = 1.0, random_state: int = None, **kwargs):
        """
        PHATE Wrapper for dimensionality reduction.

        Args:
            n_components (int): Number of dimensions to reduce to.
            knn (int): Number of nearest neighbors.
            t (str or int): Diffusion time. Can be 'auto' or a fixed integer.
            random_state (int, optional): Random seed for reproducibility.
        """
        self.n_components = n_components
        self.knn = knn
        self.t = t
        self.gamma = gamma
        self.random_state = random_state
        self.phate_params = kwargs
        self.phate_obj = None  # Placeholder for PHATE object

    def _create_phate(self):
        """Instantiate the PHATE object with specified parameters."""
        return phate.PHATE(n_components=self.n_components, knn=self.knn, t=self.t, random_state=self.random_state, **self.phate_params)

    def fit(self, data: np.ndarray):
        """Fit PHATE on the input data."""
        self.phate_obj = self._create_phate()
        self.phate_obj.fit(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using the fitted PHATE model."""
        if self.phate_obj is None:
            raise ValueError("PHATE model is not fitted yet. Call `fit()` first.")
        return self.phate_obj.transform(data)

    def fit_transform(self, data: np.ndarray, fit_idx: np.ndarray = None, transform_idx: np.ndarray = None) -> np.ndarray:
        """Fit PHATE and transform the data. If indices are provided, fit on `fit_idx` and transform `transform_idx`."""
        if fit_idx is None or transform_idx is None:
            self.phate_obj = self._create_phate()
            return self.phate_obj.fit_transform(data)

        # Fit PHATE on the selected indices
        self.phate_obj = self._create_phate()
        self.phate_obj.fit(data[fit_idx])

        # Transform both fit and transform indices
        phate_fit_emb = self.phate_obj.transform(data[fit_idx])
        phate_transform_emb = self.phate_obj.transform(data[transform_idx])

        # Concatenate transformed embeddings into original order
        phate_emb = np.zeros((data.shape[0], phate_fit_emb.shape[1]), dtype=np.float32)
        phate_emb[fit_idx] = phate_fit_emb
        phate_emb[transform_idx] = phate_transform_emb

        return phate_emb
