"""
Pluggable sampling strategies for metric evaluation.

This module provides a protocol-based interface for sampling strategies,
allowing different sampling methods to be used during metric evaluation.
The default RandomSampling preserves backward compatibility with the
original subsample_fraction behavior.

Example:
    # Direct usage
    sampler = RandomSampling()
    emb_sub, ds_sub, indices = sampler.sample(embeddings, dataset, fraction=0.1)

    # Via Hydra config
    sampling:
      _target_: manylatents.utils.sampling.StratifiedSampling
      stratify_by: population_label
      fraction: 0.1

    # Deterministic indices for reproducible comparisons
    sampler = RandomSampling(seed=42)
    indices = sampler.get_indices(n_total=1000, fraction=0.1)
    # Save indices for later comparison
    np.save('sample_indices.npy', indices)

    # Use precomputed indices
    fixed_sampler = FixedIndexSampling(indices=indices)
    emb_sub, ds_sub, _ = fixed_sampler.sample(embeddings, dataset)
"""

import logging
from copy import deepcopy
from typing import Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class SamplingStrategy(Protocol):
    """Protocol defining the interface for sampling strategies."""

    def sample(
        self,
        embeddings: np.ndarray,
        dataset: object,
        n_samples: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: int = 42,
    ) -> Tuple[np.ndarray, object, np.ndarray]:
        """
        Sample from embeddings and dataset.

        Args:
            embeddings: The embedding array to sample from.
            dataset: The dataset object with metadata attributes.
            n_samples: Absolute number of samples to take. Mutually exclusive with fraction.
            fraction: Fraction of samples to take (0.0 to 1.0). Mutually exclusive with n_samples.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (subsampled_embeddings, subsampled_dataset, indices)
            where indices are the positions of selected samples in the original data.
        """
        ...


def _compute_n_samples(
    total: int, n_samples: Optional[int], fraction: Optional[float]
) -> int:
    """Compute number of samples from either n_samples or fraction."""
    if n_samples is not None and fraction is not None:
        raise ValueError("Specify either n_samples or fraction, not both.")
    if n_samples is not None:
        if n_samples > total:
            raise ValueError(f"n_samples ({n_samples}) exceeds total ({total})")
        return n_samples
    if fraction is not None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        return int(total * fraction)
    raise ValueError("Must specify either n_samples or fraction.")


def _subsample_dataset_metadata(dataset: object, indices: np.ndarray) -> object:
    """
    Create a shallow copy of the dataset with subsampled data and metadata.

    Updates the main data array and metadata attributes (latitude, longitude,
    population_label) if present.
    """
    subsampled_ds = deepcopy(dataset)

    # Subsample the main data array (required for metrics like trustworthiness)
    if hasattr(dataset, "data"):
        subsampled_ds.data = dataset.data[indices]
        logger.debug(f"Subsampled data shape: {subsampled_ds.data.shape}")

    # Subsample metadata attributes
    if hasattr(dataset, "latitude"):
        subsampled_ds._latitude = dataset.latitude.iloc[indices]
        logger.debug(f"Subsampled latitude shape: {subsampled_ds._latitude.shape}")

    if hasattr(dataset, "longitude"):
        subsampled_ds._longitude = dataset.longitude.iloc[indices]
        logger.debug(f"Subsampled longitude shape: {subsampled_ds._longitude.shape}")

    if hasattr(dataset, "population_label"):
        pop_label = dataset.population_label
        if hasattr(pop_label, "iloc"):
            subsampled_ds._population_label = pop_label.iloc[indices]
        else:
            subsampled_ds._population_label = pop_label[indices]
        logger.debug(
            f"Subsampled population_label shape: {subsampled_ds._population_label.shape}"
        )

    return subsampled_ds


class RandomSampling:
    """
    Random sampling without replacement.

    This is the default strategy, preserving the original behavior of
    subsample_data_and_dataset().
    """

    def __init__(
        self,
        seed: int = 42,
        fraction: Optional[float] = None,
        n_samples: Optional[int] = None,
    ):
        """
        Initialize RandomSampling.

        Args:
            seed: Default random seed (can be overridden in sample()).
            fraction: Default fraction of samples (can be overridden in sample()).
            n_samples: Default number of samples (can be overridden in sample()).
        """
        self.seed = seed
        self.fraction = fraction
        self.n_samples = n_samples

    def get_indices(
        self,
        n_total: int,
        n_samples: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute sample indices without requiring data.

        Use this to precompute deterministic indices for reproducible
        comparisons across different settings or runs.

        Args:
            n_total: Total number of samples in the dataset.
            n_samples: Absolute number of samples to take.
            fraction: Fraction of samples to take.
            seed: Random seed (overrides instance seed).

        Returns:
            Sorted array of selected indices.
        """
        seed = seed if seed is not None else self.seed
        rng = np.random.default_rng(seed)
        n = _compute_n_samples(n_total, n_samples, fraction)
        indices = rng.choice(n_total, size=n, replace=False)
        return np.sort(indices)

    def sample(
        self,
        embeddings: np.ndarray,
        dataset: object,
        n_samples: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: Optional[int] = None,
        indices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, object, np.ndarray]:
        """
        Sample randomly without replacement.

        Args:
            embeddings: Embedding array to sample from.
            dataset: Dataset object with metadata.
            n_samples: Absolute number of samples (defaults to self.n_samples).
            fraction: Fraction of samples (defaults to self.fraction).
            seed: Random seed (defaults to self.seed).
            indices: Precomputed indices (bypasses random selection).

        Returns:
            Tuple of (embeddings, dataset, indices).
        """
        # Use instance defaults if not overridden
        n_samples = n_samples if n_samples is not None else self.n_samples
        fraction = fraction if fraction is not None else self.fraction
        seed = seed if seed is not None else self.seed

        if indices is not None:
            logger.info(f"RandomSampling: using precomputed indices ({len(indices)} samples)")
        else:
            indices = self.get_indices(embeddings.shape[0], n_samples, fraction, seed)
            logger.info(f"RandomSampling: {embeddings.shape[0]} -> {len(indices)} samples (seed={seed})")

        subsampled_embeddings = embeddings[indices]
        subsampled_ds = _subsample_dataset_metadata(dataset, indices)

        return subsampled_embeddings, subsampled_ds, indices


class StratifiedSampling:
    """
    Stratified sampling that preserves label distribution.

    Samples proportionally from each stratum (group) defined by a
    categorical attribute on the dataset.
    """

    def __init__(
        self,
        stratify_by: str = "population_label",
        seed: int = 42,
        fraction: Optional[float] = None,
        n_samples: Optional[int] = None,
    ):
        """
        Initialize StratifiedSampling.

        Args:
            stratify_by: Name of the dataset attribute to stratify by.
            seed: Default random seed.
            fraction: Default fraction of samples.
            n_samples: Default number of samples.
        """
        self.stratify_by = stratify_by
        self.seed = seed
        self.fraction = fraction
        self.n_samples = n_samples

    def sample(
        self,
        embeddings: np.ndarray,
        dataset: object,
        n_samples: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, object, np.ndarray]:
        """Sample proportionally from each stratum."""
        # Use instance defaults if not overridden
        n_samples = n_samples if n_samples is not None else self.n_samples
        fraction = fraction if fraction is not None else self.fraction
        seed = seed if seed is not None else self.seed

        rng = np.random.default_rng(seed)
        total = embeddings.shape[0]
        n = _compute_n_samples(total, n_samples, fraction)

        # Get stratification labels
        if not hasattr(dataset, self.stratify_by):
            logger.warning(
                f"Dataset missing '{self.stratify_by}', falling back to RandomSampling"
            )
            return RandomSampling(seed).sample(embeddings, dataset, n_samples=n)

        labels = getattr(dataset, self.stratify_by)
        if hasattr(labels, "values"):
            labels = labels.values  # Convert pandas Series to numpy

        unique_labels, label_indices = np.unique(labels, return_inverse=True)
        n_strata = len(unique_labels)

        logger.info(
            f"StratifiedSampling: {total} -> {n} samples across {n_strata} strata (seed={seed})"
        )

        # Compute samples per stratum (proportional allocation)
        counts = np.bincount(label_indices)
        proportions = counts / total
        samples_per_stratum = np.round(proportions * n).astype(int)

        # Adjust for rounding errors
        diff = n - samples_per_stratum.sum()
        if diff != 0:
            # Add/remove from largest strata
            largest_strata = np.argsort(-samples_per_stratum)
            for i in range(abs(diff)):
                idx = largest_strata[i % n_strata]
                samples_per_stratum[idx] += np.sign(diff)

        # Sample from each stratum
        selected_indices = []
        for stratum_idx, stratum_n in enumerate(samples_per_stratum):
            if stratum_n <= 0:
                continue
            stratum_mask = label_indices == stratum_idx
            stratum_positions = np.where(stratum_mask)[0]

            if stratum_n > len(stratum_positions):
                # Take all if stratum is too small
                selected_indices.extend(stratum_positions)
            else:
                sampled = rng.choice(stratum_positions, size=stratum_n, replace=False)
                selected_indices.extend(sampled)

        indices = np.sort(np.array(selected_indices))

        subsampled_embeddings = embeddings[indices]
        subsampled_ds = _subsample_dataset_metadata(dataset, indices)

        return subsampled_embeddings, subsampled_ds, indices


class FarthestPointSampling:
    """
    Farthest point sampling for maximum coverage.

    Iteratively selects points that are farthest from the already
    selected set, ensuring good coverage of the embedding space.
    Note: O(n * k) complexity where k is n_samples.
    """

    def __init__(
        self,
        seed: int = 42,
        fraction: Optional[float] = None,
        n_samples: Optional[int] = None,
    ):
        """
        Initialize FarthestPointSampling.

        Args:
            seed: Random seed for initial point selection.
            fraction: Default fraction of samples.
            n_samples: Default number of samples.
        """
        self.seed = seed
        self.fraction = fraction
        self.n_samples = n_samples

    def sample(
        self,
        embeddings: np.ndarray,
        dataset: object,
        n_samples: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, object, np.ndarray]:
        """Sample using farthest point algorithm."""
        # Use instance defaults if not overridden
        n_samples = n_samples if n_samples is not None else self.n_samples
        fraction = fraction if fraction is not None else self.fraction
        seed = seed if seed is not None else self.seed

        rng = np.random.default_rng(seed)
        total = embeddings.shape[0]
        n = _compute_n_samples(total, n_samples, fraction)

        logger.info(f"FarthestPointSampling: {total} -> {n} samples (seed={seed})")

        # Start with a random point
        indices = [rng.integers(total)]
        min_distances = np.full(total, np.inf)

        for _ in range(n - 1):
            # Update minimum distances to selected set
            last_selected = embeddings[indices[-1]]
            distances = np.linalg.norm(embeddings - last_selected, axis=1)
            min_distances = np.minimum(min_distances, distances)

            # Mask already selected
            min_distances[indices] = -np.inf

            # Select farthest point
            next_idx = np.argmax(min_distances)
            indices.append(next_idx)

        indices = np.sort(np.array(indices))

        subsampled_embeddings = embeddings[indices]
        subsampled_ds = _subsample_dataset_metadata(dataset, indices)

        return subsampled_embeddings, subsampled_ds, indices


class FixedIndexSampling:
    """
    Use precomputed indices for reproducible cross-setting comparisons.

    This strategy always uses the same indices, enabling fair comparisons
    across different algorithm settings or runs.

    Example:
        # Precompute indices once
        sampler = RandomSampling(seed=42)
        indices = sampler.get_indices(n_total=1000, fraction=0.1)
        np.save('shared_indices.npy', indices)

        # Use same indices across different settings
        fixed = FixedIndexSampling(indices=np.load('shared_indices.npy'))
        emb_sub_A, ds_sub_A, _ = fixed.sample(embeddings_A, dataset_A)
        emb_sub_B, ds_sub_B, _ = fixed.sample(embeddings_B, dataset_B)
    """

    def __init__(self, indices: Union[np.ndarray, list]):
        """
        Initialize with precomputed indices.

        Args:
            indices: Array of sample indices to use.
        """
        self.indices = np.asarray(indices)

    def get_indices(
        self,
        n_total: Optional[int] = None,
        n_samples: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Return the fixed indices (ignores all parameters)."""
        return self.indices

    def sample(
        self,
        embeddings: np.ndarray,
        dataset: object,
        n_samples: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: Optional[int] = None,
        indices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, object, np.ndarray]:
        """
        Sample using fixed indices.

        All parameters except embeddings and dataset are ignored.
        The indices passed to __init__ are always used.
        """
        # Validate indices against data size
        if self.indices.max() >= embeddings.shape[0]:
            raise ValueError(
                f"Index {self.indices.max()} out of bounds for data with "
                f"{embeddings.shape[0]} samples"
            )

        logger.info(f"FixedIndexSampling: using {len(self.indices)} precomputed indices")

        subsampled_embeddings = embeddings[self.indices]
        subsampled_ds = _subsample_dataset_metadata(dataset, self.indices)

        return subsampled_embeddings, subsampled_ds, self.indices


class BalancedLabelSampling:
    """
    Balanced sampling via median truncation.

    Reads labels from a dataset attribute (like StratifiedSampling),
    computes the median cluster size, caps all clusters at that median,
    and preserves smaller clusters fully.

    Example:
        Cluster sizes: A=1000, B=200, C=50
        Median: m=200
        Sampling: A->200, B->200, C->50
        Total = 450
    """

    def __init__(
        self,
        stratify_by: str = "population_label",
        seed: int = 42,
    ):
        self.stratify_by = stratify_by
        self.seed = seed

    def sample(
        self,
        embeddings: np.ndarray,
        dataset: object,
        n_samples: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: Optional[int] = None,
        labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, object, np.ndarray]:
        """
        Balanced sample via median truncation.

        Args:
            embeddings: Embedding array to sample from.
            dataset: Dataset object with metadata.
            n_samples: Ignored (output size determined by median truncation).
            fraction: Ignored (output size determined by median truncation).
            seed: Random seed (defaults to self.seed).
            labels: Optional explicit labels array. If provided, bypasses
                dataset attribute lookup (used by DiffusionCondensationSampling).

        Returns:
            Tuple of (embeddings, dataset, indices).
        """
        seed = seed if seed is not None else self.seed
        rng = np.random.default_rng(seed)

        # Get labels
        if labels is None:
            if not hasattr(dataset, self.stratify_by):
                raise AttributeError(
                    f"Dataset missing '{self.stratify_by}' and no explicit labels provided"
                )
            labels = getattr(dataset, self.stratify_by)
            if hasattr(labels, "values"):
                labels = labels.values  # pandas Series -> numpy

        unique_labels, label_indices = np.unique(labels, return_inverse=True)
        counts = np.bincount(label_indices)
        median_size = int(np.median(counts))

        logger.info(
            f"BalancedLabelSampling: {len(unique_labels)} clusters, "
            f"sizes {counts.tolist()}, median={median_size} (seed={seed})"
        )

        selected_indices = []
        for cluster_idx, cluster_count in enumerate(counts):
            cluster_positions = np.where(label_indices == cluster_idx)[0]
            cap = min(cluster_count, median_size)
            if cluster_count <= median_size:
                selected_indices.extend(cluster_positions)
            else:
                sampled = rng.choice(cluster_positions, size=cap, replace=False)
                selected_indices.extend(sampled)

        indices = np.sort(np.array(selected_indices))

        subsampled_embeddings = embeddings[indices]
        subsampled_ds = _subsample_dataset_metadata(dataset, indices)

        logger.info(
            f"BalancedLabelSampling: {embeddings.shape[0]} -> {len(indices)} samples"
        )

        return subsampled_embeddings, subsampled_ds, indices


class DiffusionCondensationSampling:
    """
    Unsupervised balanced sampling via diffusion condensation.

    Runs Multiscale PHATE's diffusion condensation on the raw data
    to discover cluster structure, then delegates to BalancedLabelSampling
    for median-truncated balanced subsampling.

    Requires the ``multiscale_phate`` package (optional dependency).
    """

    def __init__(
        self,
        target_clusters: Optional[int] = 10,
        landmarks: int = 2000,
        knn: int = 5,
        decay: int = 40,
        n_pca: Optional[int] = None,
        n_jobs: int = 1,
        seed: int = 42,
    ):
        self.target_clusters = target_clusters
        self.landmarks = landmarks
        self.knn = knn
        self.decay = decay
        self.n_pca = n_pca
        self.n_jobs = n_jobs
        self.seed = seed

    def _select_scale(self, model) -> int:
        """Pick the condensation scale to extract cluster labels from.

        If target_clusters is None (auto mode), uses the gradient-based
        salient level heuristic from Multiscale PHATE (model.levels[-2]).

        If target_clusters is an int, scans NxTs from index 1 onward
        (guaranteeing at least one condensation step) and picks the first
        scale where the number of unique clusters <= target_clusters.
        Falls back to the last (coarsest) scale if never satisfied.
        """
        if self.target_clusters is None:
            if len(model.levels) < 2:
                return model.levels[-1]
            return model.levels[-2]

        for i in range(1, len(model.NxTs)):
            n_unique = len(np.unique(model.NxTs[i]))
            if n_unique <= self.target_clusters:
                return i
        # Fallback: coarsest scale
        return len(model.NxTs) - 1

    def sample(
        self,
        embeddings: np.ndarray,
        dataset: object,
        n_samples: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, object, np.ndarray]:
        """
        Discover clusters via diffusion condensation and balanced-sample.

        Args:
            embeddings: Embedding array to sample from.
            dataset: Dataset object — must have .data attribute with raw data.
            n_samples: Ignored (output size determined by median truncation).
            fraction: Ignored (output size determined by median truncation).
            seed: Random seed (defaults to self.seed).

        Returns:
            Tuple of (embeddings, dataset, indices).
        """
        from multiscale_phate import Multiscale_PHATE

        seed = seed if seed is not None else self.seed

        logger.info(
            f"DiffusionCondensationSampling: fitting Multiscale PHATE on "
            f"{dataset.data.shape[0]} samples (target_clusters={self.target_clusters})"
        )

        model = Multiscale_PHATE(
            landmarks=self.landmarks,
            knn=self.knn,
            decay=self.decay,
            n_pca=self.n_pca,
            n_jobs=self.n_jobs,
            random_state=seed,
        )
        model.fit(dataset.data)

        scale_idx = self._select_scale(model)
        cluster_labels = model.NxTs[scale_idx]
        n_clusters = len(np.unique(cluster_labels))

        logger.info(
            f"DiffusionCondensationSampling: selected scale {scale_idx}, "
            f"{n_clusters} clusters"
        )

        balanced_sampler = BalancedLabelSampling(seed=seed)
        return balanced_sampler.sample(
            embeddings, dataset, labels=cluster_labels
        )
