import numpy as np
import sklearn

class PCA:
    def __init__(self, n_components: int, random_state: int = None):
        super().__init__()
        self.n_components = n_components
        self.random_state = random_state
        self.pca_obj = None 

    def _create_pca(self):
        """Instantiate the PCA object"""
        return sklearn.decomposition.PCA(n_components=self.n_components, random_state=self.random_state)

    def fit(self, data: np.ndarray):
        """Fit PCA model on input data"""
        self.pca_obj = self._create_pca()
        self.pca_obj.fit(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted PCA model"""
        if self.pca_obj is None:
            raise ValueError("PCA model is not fitted yet. Call `fit` first.")
        return self.pca_obj.transform(data)

    def fit_transform(self, data: np.ndarray, fit_idx: np.ndarray = None, transform_idx: np.ndarray = None) -> np.ndarray:
        """Fit and transform PCA"""
        if fit_idx is None or transform_idx is None:
            return self.pca_obj.fit_transform(data)
        
        self.pca_obj.fit(data[fit_idx])
        pca_fit_emb = self.pca_obj.transform(data[fit_idx])
        pca_transform_emb = self.pca_obj.transform(data[transform_idx])

        # concatenate the fit and transform data
        pca_emb = np.zeros((data.shape[0], pca_fit_emb.shape[1]))
        pca_emb[fit_idx] = pca_fit_emb
        pca_emb[transform_idx] = pca_transform_emb
        return pca_emb

    