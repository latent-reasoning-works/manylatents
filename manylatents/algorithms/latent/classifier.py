"""Supervised classifier as a LatentModule.

Trains a logistic regression classifier and exposes loadings for interpretability.
The "embedding" output is P(y=1|x), enabling downstream AUC computation.

Example:
    >>> clf = ClassifierModule(model="logistic")
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.transform(X_test)  # Returns P(y=1|x)
    >>> loadings = clf.get_loadings()  # Classifier coefficients
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import Tensor

from .latent_module_base import LatentModule, _to_numpy, _to_output


class ClassifierModule(LatentModule):
    """Supervised classifier as a LatentModule.

    Trains a classifier on (x, y) pairs and outputs predictions as "embeddings".
    Exposes get_loadings() for interpretability analysis.

    Args:
        model: Classifier type. Currently only "logistic" supported.
        n_components: Output dimension (1 for binary classification).
        max_iter: Maximum iterations for solver.
        **kwargs: Passed to LatentModule.

    Attributes:
        _clf: Fitted sklearn classifier.
    """

    def __init__(
        self,
        model: str = "logistic",
        n_components: int = 1,
        max_iter: int = 1000,
        **kwargs,
    ):
        super().__init__(n_components=n_components, **kwargs)
        self.model_type = model
        self.max_iter = max_iter
        self._clf = None

    def fit(self, x, y=None) -> None:
        """Fit classifier on input data and labels.

        Args:
            x: Input data of shape (N, D). Accepts ndarray or Tensor.
            y: Labels of shape (N,). Required for ClassifierModule.

        Raises:
            ValueError: If y is None.
        """
        if y is None:
            raise ValueError(
                "ClassifierModule requires labels (y). "
                "Pass labels via fit(x, y) or ensure your datamodule provides them."
            )

        x_np = _to_numpy(x)
        y_np = _to_numpy(y)

        if self.model_type == "logistic":
            self._clf = LogisticRegression(
                max_iter=self.max_iter,
                random_state=self.init_seed,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self._clf.fit(x_np, y_np)
        self._is_fitted = True

    def transform(self, x):
        """Return P(y=1|x) as predictions.

        Args:
            x: Input data of shape (N, D). Accepts ndarray or Tensor.

        Returns:
            Predictions of shape (N, 1) representing P(y=1|x).
            Returns same type as input (ndarray or Tensor).

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError(
                "ClassifierModule is not fitted yet. Call `fit(x, y)` first."
            )

        x_np = _to_numpy(x)

        # Get probability of positive class, return as (N, 1)
        proba = self._clf.predict_proba(x_np)[:, 1].reshape(-1, 1).astype(np.float32)
        return _to_output(proba, x)

    def get_loadings(self) -> np.ndarray:
        """Return classifier coefficients for interpretability.

        Returns:
            Coefficients of shape (1, n_features) for logistic regression.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError(
                "ClassifierModule is not fitted yet. Call `fit(x, y)` first."
            )
        return self._clf.coef_

    def get_intercept(self) -> np.ndarray:
        """Return classifier intercept.

        Returns:
            Intercept of shape (1,).

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError(
                "ClassifierModule is not fitted yet. Call `fit(x, y)` first."
            )
        return self._clf.intercept_

    def __repr__(self) -> str:
        return f"ClassifierModule(model='{self.model_type}', max_iter={self.max_iter})"
