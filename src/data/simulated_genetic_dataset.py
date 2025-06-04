import logging
import os
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
import stdpopsim
import msprime
from torch.utils.data import Dataset
import omegaconf
import pyslim
import tskit

from src.utils.data import generate_hash
from .precomputed_mixin import PrecomputedMixin
from src.utils.data import preprocess_data_matrix

logger = logging.getLogger(__name__)


class SimulatedGeneticDataset(Dataset, PrecomputedMixin):
    """
    Abstract PyTorch Dataset for simulated genetic datasets.
    """

    def __init__(self, 
                 cache_dir: str,  
                 mmap_mode: Optional[str] = None,
                 precomputed_path: Optional[str] = None
                 ) -> None:
        """
        Initialize the SimulatedGeneticDataset base class for loading or generating
        synthetic genotype data and associated metadata.

        This abstract dataset class provides a template for simulated genetic datasets
        built using population genetic simulation tools such as `msprime`. It supports 
        caching, memory-mapping, and metadata extraction, and is designed to be extended 
        by specific subclasses implementing custom simulation logic.

        Parameters
        ----------
        cache_dir : str
            Directory where simulated data and metadata will be cached.

        mmap_mode : str or None, optional
            Memory-mapping mode for reading precomputed arrays (e.g., "r", "r+", or None). 

        precomputed_path : str or None, optional
            Path to a file containing precomputed embeddings or genotype matrices.
        """
        super().__init__()

        self.cache_dir = cache_dir 
        self.mmap_mode = mmap_mode

        self.data, self.metadata = self.generate_data()

        if precomputed_path is not None:
            self.data = self.load_precomputed(precomputed_path, mmap_mode=mmap_mode)

        if self.data is None:
            raise ValueError("No data source found: either failed to generate or precomputed embeddings are missing.")

        self._latitude = self.extract_latitude()
        self._longitude = self.extract_longitude()
        self._population_label = self.extract_population_label()
        self._geographic_preservation_indices = self.extract_geographic_preservation_indices()
        self.admixture_ratios = self.load_admixture_ratios()

    def generate_data(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Generate simulated genotype matrix and associated metadata."""
        self.demography = self._load_demographic_model()
        self.samples, self.idx_to_name = self._define_sample_map(self.pop_sizes)
        self.ts_ancestry = self._simulate_ancestry()
        self.ts = self._simulate_mutations()

        genotypes = self._filter_variants(self.ts)
        genotypes = self._preprocess(genotypes)
        metadata = self._extract_metadata(self.ts)
        return genotypes, metadata

    def _define_sample_map(self, sizes):
        if isinstance(sizes, int):
            pop_sizes = {pop.name: sizes for pop in self.demography.populations}
        
        # hydra will send as omegaconf.listconfig.ListConfig NOT list
        elif isinstance(sizes, list) or isinstance(sizes, omegaconf.listconfig.ListConfig):
            assert len(sizes) == len(self.demography.populations)
            pop_sizes = {pop.name: 
                         sizes[i] for i, pop in enumerate(self.demography.populations)}
        else:
            raise Exception('Invalid type for sizes. Should be list or int')
        idx_to_name = [pop.name for pop in self.demography.populations]
        return pop_sizes, idx_to_name

    def _simulate_ancestry(self):
        return msprime.sim_ancestry(
            samples=self.samples,
            ploidy=self.ploidy,
            sequence_length=self.sequence_length,
            recombination_rate=self.recombination_rate,
            demography=self.demography,
            random_seed=self.random_state,
        )

    def _simulate_mutations(self):
        return msprime.sim_mutations(
            self.ts_ancestry,
            rate=self.mutation_rate,
            random_seed=self.random_state
        )

    def _filter_variants(self, ts):
        G = ts.genotype_matrix()
        ac = G.sum(axis=1)
        n_samples = G.shape[1]
        mac = np.minimum(ac, n_samples - ac)
        keep = mac >= self.mac_threshold
        G = G[keep]

        if self.num_variants:
            G = G[:self.num_variants]

        return G.T  # shape: (n_samples, n_variants)
    
    def _preprocess(self, genotypes):
        # just assume we are fitting on all of the data
        normalized_matrix = preprocess_data_matrix(genotypes_array=genotypes, 
                                                   fit_idx=np.arange(len(genotypes)), 
                                                   trans_idx=np.arange(len(genotypes)))
        return normalized_matrix

    def _extract_metadata(self, ts):
        samples = ts.samples()
        node_table = ts.tables.nodes
        pop_ids = node_table.population[samples]
        times = node_table.time[samples]

        # convert pop_ids to named categorical variable
        named_pops = np.array(self.idx_to_name)[pop_ids]  # vectorized
        categorical_pops = pd.Categorical(named_pops, categories=self.idx_to_name)

        return pd.DataFrame({
            "sample_id": samples,
            "time": times,
            "Population": categorical_pops
        })

    @abstractmethod
    def _load_demographic_model(self, **kwargs):
        """loads demographic model"""
        pass

    @abstractmethod
    def extract_latitude(self) -> pd.Series:
        """Extract latitude values for each sample."""
        pass

    @abstractmethod
    def extract_longitude(self) -> pd.Series:
        """Extract longitude values for each sample."""
        pass

    @abstractmethod
    def extract_population_label(self) -> pd.Series:
        """Extract population labels or ancestry categories."""
        pass

    @abstractmethod
    def extract_geographic_preservation_indices(self) -> np.ndarray:
        """Extract a preservation or grouping index (e.g., for spatial structure)."""
        pass

    @abstractmethod
    def load_admixture_ratios(self) -> dict:
        """Return admixture ratios or ancestral proportions if applicable."""
        pass

    @abstractmethod
    def get_labels(self, label_col: str = "Population") -> np.ndarray:
        """Return an array of sample-level labels used for evaluation or supervision."""
        pass

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        sample = self.data[idx]
        metadata_row = self.metadata.iloc[idx].to_dict()
        metadata_row = {k.strip(): v for k, v in metadata_row.items()}
        return {"data": sample, "metadata": metadata_row}

    @property
    def latitude(self) -> pd.Series:
        return self._latitude

    @property
    def longitude(self) -> pd.Series:
        return self._longitude

    @property
    def population_label(self) -> pd.Series:
        return self._population_label


class StdPopSimDataHumanDemoModel(SimulatedGeneticDataset):
    def __init__(
        self,
        cache_dir,
        mmap_mode=None,
        precomputed_path=None,
        pop_sizes=500,
        demographic_model='OutOfAfrica_3G09',
        num_variants=1000,
        mac_threshold=20,
        mutation_rate=1.25e-8,
        recombination_rate=1e-8,
        sequence_length=2e7,
        ploidy=2,
        random_state=42
    ):
        """
        Initialize the StdPopSimDataHumanDemoModel with parameters for generating 
        synthetic genetic data based on standardized human demographic models.

        This class simulates genotype matrices and associated metadata using 
        `msprime` and `stdpopsim`, providing realistic population structure 
        informed by empirical models such as OutOfAfrica_3G09.

        Parameters
        ----------
        cache_dir : str
            Directory for caching preprocessed data.

        mmap_mode : str or None, optional
            Memory-mapping mode for loading precomputed data (e.g., 'r' or 'r+').
            Useful for large datasets to avoid loading the full array into memory.

        precomputed_path : str or None, optional
            Optional path to precomputed embeddings or genotype arrays. If provided,
            data loading will use this path instead of running the full 
            simulation pipeline.

        pop_sizes : int, default=500
            Number of diploid individuals to sample per population defined in the
            demographic model.

        demographic_model : str, default='OutOfAfrica_3G09'
            Identifier for a standard human demographic model provided by `stdpopsim`.
            Examples include 'OutOfAfrica_3G09', 
            'Africa_1T12', or 'AmericanAdmixture_4B11'.

        num_variants : Union[int, None], default=1000
            Number of filtered variants to retain after applying 
            the minor allele count (MAC) threshold.
            None = keep all variants

        mac_threshold : int, default=20
            Minimum minor allele count required for a variant to be retained. 
            Filters out ultra-rare variants to better match empirical GWAS-style data.

        mutation_rate : float, default=1.25e-8
            Mutation rate per base pair per generation. 
            Set to reflect realistic human mutation rates.

        recombination_rate : float, default=1e-8
            Recombination rate per base pair per generation. 
            Used to simulate ancestry with recombination.

        sequence_length : float, default=2e7
            Total length of the simulated genomic region (in base pairs). 
            Larger regions yield more variants.

        ploidy : int, default=2
            Number of genome copies per individual (typically 2 for diploid organisms).

        random_state : int, default=42
            Seed for the random number generator used in both 
            ancestry and mutation simulations.
        """
        self.pop_sizes = pop_sizes
        self.demographic_model = demographic_model
        self.num_variants = num_variants
        self.mac_threshold = mac_threshold
        self.mutation_rate = mutation_rate
        self.recombination_rate = recombination_rate
        self.sequence_length = sequence_length
        self.ploidy = ploidy
        self.random_state = random_state

        super().__init__(
            cache_dir=cache_dir,
            mmap_mode=mmap_mode,
            precomputed_path=precomputed_path
        )
        
    def _load_demographic_model(self) -> msprime.demography.Demography:
        
        # load species
        self.species = stdpopsim.get_species("HomSap")

        # enumerate fixed models
        self._possible_models = []
        for species in self.species.demographic_models:
            self._possible_models.append(species.id)
        
        # check that model is defined
        if self.demographic_model in self._possible_models:
            model = self.species.get_demographic_model(self.demographic_model)
            return model.model
        else:
            raise Exception("Could not load {}. Not a valid model".format(self.demographic_model))
        raise Exception("No demographic_model in kwargs!")

    def extract_latitude(self) -> pd.Series:
        # Placeholder; update if you add lat/lon metadata
        return pd.Series([np.nan] * len(self.metadata))

    def extract_longitude(self) -> pd.Series:
        return pd.Series([np.nan] * len(self.metadata))

    def extract_population_label(self) -> pd.Series:
        return self.metadata["Population"]

    def extract_geographic_preservation_indices(self) -> np.ndarray:
        # Placeholder; update if applicable
        return np.zeros(len(self.metadata), dtype=int)

    def load_admixture_ratios(self) -> dict:
        # Placeholder
        return {}

    def get_labels(self, label_col: str = "Population") -> np.ndarray:
        return self.metadata[label_col].values

    def get_gt_dists(self, k=10):
        """
        Approximate geodesic distances using shortest paths over a k-NN graph
        derived from the divergence matrix.

        Parameters
        ----------
        ts : tskit.TreeSequence
        k : int
            Number of neighbors to keep for each sample (sparsification).

        Returns
        -------
        np.ndarray
            Geodesic distance matrix approximating the genetic manifold.
        """
        from sklearn.neighbors import NearestNeighbors

        D = self.ts.divergence_matrix()
        n = D.shape[0]

        # Build a sparse k-NN graph from divergence
        nn = NearestNeighbors(n_neighbors=k + 1, metric="precomputed")
        nn.fit(D)
        distances, indices = nn.kneighbors(D)

        # Create sparse matrix
        graph = sp.lil_matrix((n, n))
        for i in range(n):
            for j_idx, j in enumerate(indices[i][1:]):  # skip self
                dist = distances[i][j_idx + 1]
                graph[i, j] = dist
                graph[j, i] = dist  # make symmetric

        # Compute geodesic distances
        geo_dists = shortest_path(csgraph=graph.tocsr(), directed=False)
        return geo_dists

    
class CustomAdmixedModel(StdPopSimDataHumanDemoModel):
    def __init__(
        self,
        cache_dir,
        mmap_mode=None,
        precomputed_path=None,
        pop_sizes=500,
        demographic_model_path='OutOfAfrica_3G09',
        num_variants=1000,
        mac_threshold=20,
        mutation_rate=1.25e-8,
        recombination_rate=1e-8,
        sequence_length=2e7,
        ploidy=2,
        random_state=42
    ):
        """
        Initialize the StdPopSimDataHumanDemoModel with parameters for generating 
        synthetic genetic data based on standardized human demographic models.

        This class simulates genotype matrices and associated metadata using 
        `msprime` and `stdpopsim`, providing realistic population structure 
        informed by empirical models such as OutOfAfrica_3G09.

        Parameters
        ----------
        cache_dir : str
            Directory for caching preprocessed data.

        mmap_mode : str or None, optional
            Memory-mapping mode for loading precomputed data (e.g., 'r' or 'r+').
            Useful for large datasets to avoid loading the full array into memory.

        precomputed_path : str or None, optional
            Optional path to precomputed embeddings or genotype arrays. If provided,
            data loading will use this path instead of running the full 
            simulation pipeline.

        pop_sizes : int, default=500
            Number of diploid individuals to sample per population defined in the
            demographic model.

        demographic_model : str, default='OutOfAfrica_3G09'
            Identifier for a standard human demographic model provided by `stdpopsim`.
            Examples include 'OutOfAfrica_3G09', 
            'Africa_1T12', or 'AmericanAdmixture_4B11'.

        num_variants : Union[int, None], default=1000
            Number of filtered variants to retain after applying 
            the minor allele count (MAC) threshold.
            None = keep all variants

        mac_threshold : int, default=20
            Minimum minor allele count required for a variant to be retained. 
            Filters out ultra-rare variants to better match empirical GWAS-style data.

        mutation_rate : float, default=1.25e-8
            Mutation rate per base pair per generation. 
            Set to reflect realistic human mutation rates.

        recombination_rate : float, default=1e-8
            Recombination rate per base pair per generation. 
            Used to simulate ancestry with recombination.

        sequence_length : float, default=2e7
            Total length of the simulated genomic region (in base pairs). 
            Larger regions yield more variants.

        ploidy : int, default=2
            Number of genome copies per individual (typically 2 for diploid organisms).

        random_state : int, default=42
            Seed for the random number generator used in both 
            ancestry and mutation simulations.
        """
        self.pop_sizes = pop_sizes
        self.demographic_model_path = demographic_model_path
        self.num_variants = num_variants
        self.mac_threshold = mac_threshold
        self.mutation_rate = mutation_rate
        self.recombination_rate = recombination_rate
        self.sequence_length = sequence_length
        self.ploidy = ploidy
        self.random_state = random_state

        super().__init__(
            cache_dir=cache_dir,
            mmap_mode=mmap_mode,
            precomputed_path=precomputed_path
        )

    """
    Same but now we use a custom demographic model
    """
    def generate_data(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Generate simulated genotype matrix and associated metadata."""
        self.demography = self._load_demographic_model()
        self.samples, self.idx_to_name = self._define_sample_map(self.pop_sizes)
        self.ts = pyslim.tskit.load(self.demographic_model_path)

        # Get genotype matrix (haploid, shape = num_sites × 2*num_individuals)
        G = self.ts.genotype_matrix()

        import pdb
        pdb.set_trace()

        # Recapitate to add coalescent history (required for nonWF)
        ts_recap = self.ts.recapitate(recombination_rate=1e-8, Ne=10000)

        # Add mutations after recapitation
        ts_mut = msprime.sim_mutations(ts_recap, rate=1e-7, random_seed=42)

        print(f"{ts_mut.num_sites} sites, {ts_mut.num_mutations} mutations")
        
        # usual after?

        # Convert to diploid (reshape to: num_sites × num_individuals × 2)
        G_reshaped = G.reshape(G.shape[0], -1, 2)

        # Sum haplotypes → genotypes 0/1/2
        diploid_genotypes = G_reshaped.sum(axis=-1).T

        diploid_genotypes = self._preprocess(diploid_genotypes)

        metadata = self._extract_metadata(self.ts)

        return diploid_genotypes, metadata

    def _extract_metadata(self, ts):
        # Load SLiM-specific info
        assert isinstance(ts, tskit.trees.TreeSequence)

        samples = ts.samples()
        node_table = ts.tables.nodes

        # Group samples into diploid individuals (2 haploids per individual)
        assert len(samples) % 2 == 0, "Expected even number of haploid samples (diploids)"
        sample_pairs = samples.reshape(-1, 2)

        # Use the first haploid in each pair to identify the individual
        pop_ids = node_table.population[sample_pairs[:, 0]]
        times = node_table.time[sample_pairs[:, 0]]
        individual_ids = node_table.individual[sample_pairs[:, 0]]

        # Convert population IDs to categorical labels
        named_pops = np.array(self.idx_to_name)[pop_ids]
        categorical_pops = pd.Categorical(named_pops, categories=self.idx_to_name)

        # Extract ancestry vectors from individuals
        individuals = ts.individuals()
        ancestry_vectors = []
        for ind_id in individual_ids:
            ind = individuals[ind_id]
            ancestry = ind.metadata.get("ancestry", None)
            if ancestry is None:
                ancestry = [np.nan] * len(self.idx_to_name)
            ancestry_vectors.append(np.array(ancestry))

        return pd.DataFrame({
            "individual_id": np.arange(len(sample_pairs)),
            "haploids": list(sample_pairs),
            "time": times,
            "Population": categorical_pops,
            "ancestry": ancestry_vectors
        })