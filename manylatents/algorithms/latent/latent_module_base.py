from abc import ABC, abstractmethod
import numpy as np

from torch import Tensor


class LatentModule(ABC):
    def __init__(self, n_components: int = 2, init_seed: int = 42,
                 backend: str | None = None, device: str | None = None, **kwargs):
        """Base class for latent modules (DR, clustering, etc.)."""
        self.n_components = n_components
        self.init_seed = init_seed
        self.backend = backend
        self.device = device
        # Flexible handling: if datamodule is passed, store it as a weak port
        self.datamodule = kwargs.pop('datamodule', None)
        # Ignore any other unexpected kwargs to maintain compatibility
        self._is_fitted = False

    @abstractmethod
    def fit(self, x: Tensor, y: Tensor | None = None) -> None:
        """Fit the module on data.

        Args:
            x: Input data of shape (N, D).
            y: Optional labels of shape (N,) for supervised methods.
               Ignored by unsupervised modules.
        """
        pass

    @abstractmethod
    def transform(self, x: Tensor) -> Tensor:
        pass

    def fit_transform(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        self.fit(x, y)
        return self.transform(x)

    def kernel_matrix(self, ignore_diagonal: bool = False) -> np.ndarray:
        """
        Return the kernel matrix (similarity matrix) used by the algorithm.

        The kernel matrix is a symmetric N×N matrix where entry (i,j) represents
        the similarity between samples i and j. This is the raw kernel before
        any normalization is applied.

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.

        Returns:
            N×N numpy array representing the kernel matrix.

        Raises:
            NotImplementedError: If the algorithm does not expose a kernel matrix.
            RuntimeError: If called before fitting the model.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose a kernel_matrix. "
            "This may be because the algorithm does not use a kernel-based approach."
        )

    def affinity_matrix(self, ignore_diagonal: bool = False, use_symmetric: bool = False) -> np.ndarray:
        """
        Return the affinity matrix (normalized transition matrix) used by the algorithm.

        The affinity matrix is derived from the kernel matrix through normalization
        (e.g., row normalization, diffusion normalization). It often represents
        transition probabilities or diffusion operators.

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.
            use_symmetric: If True, return a symmetric version of the affinity matrix
                          (if available). For methods like diffusion maps, this returns
                          the symmetric diffusion operator with guaranteed positive
                          eigenvalues. Default False.

        Returns:
            N×N numpy array representing the affinity matrix.

        Raises:
            NotImplementedError: If the algorithm does not expose an affinity matrix.
            RuntimeError: If called before fitting the model.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose an affinity_matrix. "
            "This may be because the algorithm does not use a kernel-based approach."
        )

    def affinity_tensor(self) -> 'torch.Tensor':
        """Return affinity matrix as a torch.Tensor.

        When the TorchDR backend is active, returns the GPU tensor directly.
        Otherwise, converts the numpy affinity matrix.

        Returns:
            torch.Tensor: Affinity matrix on compute device.
        """
        import torch

        if self.backend == "torchdr" and hasattr(self, 'model') and hasattr(self.model, 'affinity_in_'):
            return self.model.affinity_in_.detach()
        return torch.from_numpy(
            self.affinity_matrix(use_symmetric=True).astype('float32')
        )
