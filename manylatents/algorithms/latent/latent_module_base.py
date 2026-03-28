import warnings
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
from torch import Tensor

ArrayLike = Union[np.ndarray, Tensor]


def _to_numpy(x):
    """Convert Tensor or ndarray to ndarray."""
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_output(result, input_ref):
    """Match output type to input type. For transform/fit_transform only."""
    if isinstance(input_ref, Tensor):
        if isinstance(result, np.ndarray):
            return torch.tensor(result, device=input_ref.device, dtype=input_ref.dtype)
        return result  # already a Tensor
    else:
        if isinstance(result, Tensor):
            return result.detach().cpu().numpy()
        return np.asarray(result)


class LatentModule(ABC):
    def __init__(self, n_components: int = 2, init_seed: int = 42,
                 backend: str | None = None, device: str | None = None,
                 neighborhood_size: int | None = None, **kwargs):
        """Base class for latent modules (DR, clustering, etc.)."""
        self.n_components = n_components
        self.init_seed = init_seed
        self.backend = backend
        self.device = device
        self.neighborhood_size = neighborhood_size
        # Flexible handling: if datamodule is passed, store it as a weak port
        self.datamodule = kwargs.pop('datamodule', None)
        # Ignore any other unexpected kwargs to maintain compatibility
        self._is_fitted = False

    @abstractmethod
    def fit(self, x: ArrayLike, y: ArrayLike | None = None) -> None:
        """Fit the module on data.

        Args:
            x: Input data of shape (N, D). Accepts ndarray or Tensor.
            y: Optional labels of shape (N,) for supervised methods.
               Ignored by unsupervised modules.
        """
        pass

    @abstractmethod
    def transform(self, x: ArrayLike) -> ArrayLike:
        pass

    def fit_transform(self, x: ArrayLike, y: ArrayLike | None = None) -> ArrayLike:
        self.fit(x, y)
        return self.transform(x)

    def kernel(self, ignore_diagonal: bool = False) -> np.ndarray:
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
            f"{self.__class__.__name__} does not expose a kernel. "
            "This may be because the algorithm does not use a kernel-based approach."
        )

    def affinity(self, ignore_diagonal: bool = False, use_symmetric: bool = False) -> np.ndarray:
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
            f"{self.__class__.__name__} does not expose an affinity. "
            "This may be because the algorithm does not use a kernel-based approach."
        )

    def adjacency(self, ignore_diagonal: bool = False) -> np.ndarray:
        """Return the adjacency matrix (binary connectivity) of the algorithm's graph.

        The adjacency matrix is a binary (0/1) matrix where entry (i,j) is 1 iff
        nodes i and j are connected. Unlike kernel (weighted similarity)
        or affinity (transition probabilities), this is unweighted.

        The matrix may be M×M where M differs from the number of input samples N
        (e.g., Reeb graph nodes vs data points).

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.

        Returns:
            Binary numpy array representing graph connectivity.

        Raises:
            NotImplementedError: If the algorithm does not expose an adjacency matrix.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose an adjacency. "
            "This may be because the algorithm does not produce a graph structure."
        )

    # Deprecated aliases — old _matrix suffix names
    def kernel_matrix(self, ignore_diagonal: bool = False) -> np.ndarray:
        """Deprecated: use kernel() instead."""
        warnings.warn(
            "kernel_matrix() is deprecated, use kernel() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.kernel(ignore_diagonal=ignore_diagonal)

    def affinity_matrix(self, ignore_diagonal: bool = False, use_symmetric: bool = False) -> np.ndarray:
        """Deprecated: use affinity() instead."""
        warnings.warn(
            "affinity_matrix() is deprecated, use affinity() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.affinity(ignore_diagonal=ignore_diagonal, use_symmetric=use_symmetric)

    def adjacency_matrix(self, ignore_diagonal: bool = False) -> np.ndarray:
        """Deprecated: use adjacency() instead."""
        warnings.warn(
            "adjacency_matrix() is deprecated, use adjacency() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.adjacency(ignore_diagonal=ignore_diagonal)

    def extra_outputs(self) -> dict:
        """Collect extra outputs from the algorithm for attachment to LatentOutputs.

        Base implementation collects trajectories, affinity, adjacency,
        and kernel when available. Subclasses can override to add their own.

        Returns:
            dict of collected outputs (empty if nothing available).
        """
        extras = {}

        # Trajectories (stored as attribute, not a method)
        traj = getattr(self, "trajectories", None)
        if traj is not None:
            if isinstance(traj, Tensor):
                traj = traj.detach().cpu().numpy()
            extras["trajectories"] = traj

        # Matrix outputs via methods (short keys)
        for name, method_name in [
            ("affinity", "affinity"),
            ("adjacency", "adjacency"),
            ("kernel", "kernel"),
        ]:
            try:
                val = getattr(self, method_name)()
                extras[name] = val
            except (NotImplementedError, AttributeError, RuntimeError):
                pass

        return extras

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
            self.affinity(use_symmetric=True).astype('float32')
        )
