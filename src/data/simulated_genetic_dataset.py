import logging
import os
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple
from tqdm import tqdm
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

from .precomputed_mixin import PrecomputedMixin
from src.utils.data import preprocess_data_matrix

logger = logging.getLogger(__name__)


def trace_ancestry(ts, tracing_time=100, num_trees_to_sample=500):
    node_pop = ts.tables.nodes.population
    all_tree_indices = np.arange(ts.num_trees)
    sampled_indices = np.random.choice(all_tree_indices, size=num_trees_to_sample, replace=False)

    ancestry = np.zeros((ts.num_individuals, len(ts.populations())), dtype=float)

    for idx in tqdm(sampled_indices, desc="Assigning ancestry ratio using sampled trees"):
        tree = ts.at(idx)
        for i, ind in enumerate(ts.individuals()):
            for node in ind.nodes:
                anc_time = tree.get_time(node)
                current = node
                while anc_time < tracing_time and tree.parent(current) != tskit.NULL:
                    current = tree.parent(current)
                    anc_time = tree.time(current)
                pop = node_pop[current]
                ancestry[i, pop] += tree.span

    ancestry /= 2 * np.sum([ts.at(idx).span for idx in sampled_indices])

    # Label populations
    pop_names = [p.metadata.get("name", f"pop{p.id}") for p in ts.populations()]
    df = pd.DataFrame(ancestry, columns=pop_names)
    return df


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

        self.data, self.metadata, self.gt_admixture_ratios = self.generate_data()

        if precomputed_path is not None:
            self.data = self.load_precomputed(precomputed_path, mmap_mode=mmap_mode)

        if self.data is None:
            raise ValueError("No data source found: either failed to generate or precomputed embeddings are missing.")

        self._latitude = self.extract_latitude()
        self._longitude = self.extract_longitude()
        self._population_label = self.extract_population_label()
        self._geographic_preservation_indices = self.extract_geographic_preservation_indices()
        self.admixture_ratios = self.load_admixture_ratios()

    def generate_data(self) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
        """Generate simulated genotype matrix and associated metadata."""
        import pdb
        pdb.set_trace()
        self.demography = self._load_demographic_model()
        self.samples, self.idx_to_name = self._define_sample_map(self.pop_sizes)
        self.ts_ancestry = self._simulate_ancestry()
        self.ts = self._simulate_mutations()
        genotypes = self._get_diploid_genotype_matrix()
        genotypes = self._filter_variants(genotypes)
        genotypes = self._preprocess(genotypes)

        metadata_df = self.extract_individual_metadata()
        ancestry_df = trace_ancestry(self.ts, tracing_time=100, num_trees_to_sample=500)

        return genotypes, metadata_df, ancestry_df

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
            random_seed=self.random_state,
            model=msprime.JC69()
        )

    def _get_diploid_genotype_matrix(self):
        haploid_G = self.ts.genotype_matrix().T
        haploid_G = np.minimum(haploid_G, 1)

        indiv_ids = self.ts.tables.nodes.individual[:]
        diploid_inds = np.unique(indiv_ids[indiv_ids >= 0])
        num_inds = len(diploid_inds)
        num_sites = haploid_G.shape[1]

        diploid_G = np.zeros((num_inds, num_sites), dtype=int)

        for k, ind_id in enumerate(diploid_inds):
            node_ids = self.ts.individual(ind_id).nodes
            diploid_G[k, :] = haploid_G[node_ids[0], :] + haploid_G[node_ids[1], :]

        return diploid_G

    def _filter_variants(self, G_raw):
        # transpose again to be consistent
        G_raw = G_raw.T
        ac = G_raw.sum(axis=1)
        n_samples = G_raw.shape[1]
        mac = np.minimum(ac, n_samples - ac)
        keep = mac >= self.mac_threshold
        G = G_raw[keep]

        if self.num_variants:
            G = G[:self.num_variants]

        return G.T  # shape: (n_samples, n_variants)
    
    def _preprocess(self, genotypes):
        # just assume we are fitting on all of the data
        normalized_matrix = preprocess_data_matrix(genotypes_array=genotypes, 
                                                   fit_idx=np.arange(len(genotypes)), 
                                                   trans_idx=np.arange(len(genotypes)))
        return normalized_matrix

    def extract_individual_metadata(self):
        rows = []
        for ind in self.ts.individuals():
            if len(ind.nodes) != 2:
                continue
            node0 = self.ts.node(ind.nodes[0])
            rows.append({
                "individual_id": ind.id,
                "Population_id": node0.population,
                "time": node0.time,
                "nodes": ind.nodes,
            })
        
        df = pd.DataFrame(rows)

        # Add named categorical variable
        df["Population"] = pd.Categorical(df["Population_id"]).rename_categories(self.idx_to_name)

        return df
    
    def load_admixture_ratios(self) -> dict:
        """Return admixture ratios or ancestral proportions if applicable."""
        return {len(self.idx_to_name): self.gt_admixture_ratios}

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
        Compute diploid-level geodesic distances based on pairwise divergence
        between individuals (averaged over their two nodes).
        """
        from sklearn.neighbors import NearestNeighbors

        # Note the ground truth distance should be the average of TMRCA (avg over each haploid to get per-diploid score)
        # See implementation here: https://tskit.dev/tskit/docs/latest/_modules/tskit/trees.html
        D_node = self.ts.divergence_matrix()
        
        # Map individuals to their node pairs
        indiv_nodes = [ind.nodes for ind in self.ts.individuals() if len(ind.nodes) == 2]
        n = len(indiv_nodes)
        D_indiv = np.zeros((n, n))

        # Average divergence across the 4 node combinations (2x2)
        for i in range(n):
            for j in range(i, n):
                node_i = indiv_nodes[i]
                node_j = indiv_nodes[j]
                divs = [
                    D_node[node_i[0], node_j[0]],
                    D_node[node_i[0], node_j[1]],
                    D_node[node_i[1], node_j[0]],
                    D_node[node_i[1], node_j[1]],
                ]
                avg_div = np.mean(divs)
                D_indiv[i, j] = avg_div
                D_indiv[j, i] = avg_div

        # Build sparse k-NN graph
        nn = NearestNeighbors(n_neighbors=k + 1, metric="precomputed")
        nn.fit(D_indiv)
        distances, indices = nn.kneighbors(D_indiv)

        graph = sp.lil_matrix((n, n))
        for i in range(n):
            for j_idx, j in enumerate(indices[i][1:]):
                dist = distances[i][j_idx + 1]
                graph[i, j] = dist
                graph[j, i] = dist

        geo_dists = shortest_path(csgraph=graph.tocsr(), directed=False)
        return geo_dists

    
class CustomAdmixedModel(StdPopSimDataHumanDemoModel):
    def __init__(
        self,
        cache_dir,
        mmap_mode=None,
        precomputed_path=None,
        pop_sizes=500,
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
        self.num_variants = num_variants
        self.mac_threshold = mac_threshold
        self.mutation_rate = mutation_rate
        self.recombination_rate = recombination_rate
        self.sequence_length = sequence_length
        self.ploidy = ploidy
        self.random_state = random_state

        self.admixture_matrix = np.array([[0.0, 3.e-05	,3.e-05],
                                            [3.e-05, 0.0, 0],
                                            [0, 3.e-05, 0.0]])
        self.admixture_start_time = 10
        self.admixture_end_time = 0
        self.ancestral_name = "ANC"
        self.root_size = 1e4
        self.growth_rate = 0.004
        self.pop_size = 1e4
        self.num_pops = len(self.pop_sizes)
        self.split_time = 1000

        super().__init__(
            cache_dir=cache_dir,
            mmap_mode=mmap_mode,
            precomputed_path=precomputed_path
        )

    """
    Same but now we use a custom demographic model
    """
    def _load_demographic_model(self) -> msprime.demography.Demography:
        """
        Create a demography with K populations split from one ancestor and with fixed admixture 
        proportions starting at admixture_start_time and stopping at admixture_end_time.

        Parameters:
        - num_pops (int): Number of derived populations.
        - split_time (float): Time at which the ancestral population splits into K populations.
        - admixture_matrix (np.ndarray): K x K matrix of migration/admixture proportions.
        - admixture_start_time (float): Time when admixture begins (in generations ago).
        - admixture_end_time (float): Time when admixture ends (0 = present).
        - ancestral_name (str): Name of the root population.
        - root_size (float): Size of ancestral population.
        - pop_size (float): Size of each derived population.

        Returns:
        - msprime.Demography object
        """
        assert self.admixture_matrix.shape == (self.num_pops, self.num_pops), "Admixture matrix must be KxK. Got {}".format(self.admixture_matrix.shape)

        demog = msprime.Demography()

        # Add ancestral population
        demog.add_population(name=self.ancestral_name, initial_size=self.root_size)

        # Add K derived populations
        pop_names = [f"POP_{i}" for i in range(self.num_pops)]
        for name in pop_names:
            demog.add_population(name=name, initial_size=self.pop_size, growth_rate=self.growth_rate)

        # Split K populations from ancestor
        demog.add_population_split(time=self.split_time, derived=pop_names, ancestral=self.ancestral_name)

        # Ensure zero migration to start with (optional, msprime defaults to 0)
        for i in range(self.num_pops):
            for j in range(self.num_pops):
                if i != j:
                    demog.set_migration_rate(source=pop_names[i], dest=pop_names[j], rate=0.0)

        # Turn on migration at admixture_start_time
        for i in range(self.num_pops):
            for j in range(self.num_pops):
                if i != j and self.admixture_matrix[i, j] > 0:
                    demog.add_migration_rate_change(
                        time=self.admixture_start_time,
                        source=pop_names[i],
                        dest=pop_names[j],
                        rate=self.admixture_matrix[i, j]
                    )

        # Optional: turn off migration again at admixture_end_time
        if self.admixture_end_time > 0:
            for i in range(self.num_pops):
                for j in range(self.num_pops):
                    if i != j and self.admixture_matrix[i, j] > 0:
                        demog.add_migration_rate_change(
                            time=self.admixture_end_time,
                            source=pop_names[i],
                            dest=pop_names[j],
                            rate=0.0
                        )
        
        demog.sort_events()

        return demog