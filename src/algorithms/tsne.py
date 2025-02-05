import numpy as np
import sklearn

class TSNE:
    def __init__(self, n_components: int = 2, random_state: int = None, perplexity: float = 30.0, **kwargs):
        """
        t-SNE Wrapper for dimensionality reduction.

        Args:
            n_components (int): Number of dimensions to reduce to.
            random_state (int, optional): Random seed for reproducibility.
            perplexity (float): Perplexity parameter for t-SNE.
            **kwargs: Additional t-SNE parameters (e.g., learning_rate, metric, etc.).
        """
        self.n_components = n_components
        self.random_state = random_state
        self.perplexity = perplexity
        self.tsne_obj = None 
        self.tsne_params = kwargs 

    def _create_tsne(self):
        """Instantiate the t-SNE object with stored parameters."""
        return sklearn.manifold.TSNE(n_components=self.n_components, 
                                     random_state=self.random_state, 
                                     perplexity=self.perplexity,
                                     **self.tsne_params)

    def fit(self, data: np.ndarray):
        """
        t-SNE does not support separate fit and transform, this function do nothering but create a t-SNE object.
        """
        self.tsne_obj = self._create_tsne()

    def transform(self, data: np.ndarray) -> np.ndarray:   
        """
        There's no transform method in TSNE, but here is to make it consistent with other methods. 
        """
        if self.tsne_obj is None:
            raise ValueError("t-SNE model is not created. Call `fit()` first.")
        return self.tsne_obj.fit_transform(data)  # t-SNE requires fitting every time

    def fit_transform(self, data: np.ndarray, fit_idx: np.ndarray = None, transform_idx: np.ndarray = None) -> np.ndarray:
        """
        Fit and transform t-SNE to the data. t-SNE does not support extend transform separately. 
        `fit_idx` and `transform_idx` won't be used for TSNE
        """
        self.tsne_obj = self._create_tsne()
        return self.tsne_obj.fit_transform(data)  

