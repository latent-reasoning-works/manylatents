import torch
from typing import Union, List
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from src.data.simulated_genetic_dataset import CustomAdmixedModel

class CustomAdmixedModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for msprime-based simulated human genetic data.
    """

    def __init__(
        self,
        batch_size: int = 128,
        test_split: float = 0.2,
        num_workers: int = 0,
        cache_dir: str = "./cache",
        pop_sizes: Union[List[int], int] = 500,
        num_variants: Union[int, None] = 1000,
        mac_threshold: int = 20,
        sequence_length: float = 2e7,
        mutation_rate: float = 1.25e-8,
        recombination_rate: float = 1e-8,
        random_state: int = 42,
        mode: str = "split",
    ):
        """
        Initialize the SimulatedGeneticDataModule with configuration parameters 
        for generating and loading synthetic human genetic datasets.

        This module wraps the `StdPopSimDataHumanDemoModel` dataset into a 
        PyTorch Lightning DataModule interface, enabling modular simulation and 
        efficient batching of realistic genotype matrices with population structure 
        based on human demographic models.

        Parameters
        ----------
        batch_size : int, default=128
            Number of samples per batch used in training and validation data loaders.

        test_split : float, default=0.2
            Fraction of the dataset to allocate to the test set. 
            Only used when `mode='split'`.

        num_workers : int, default=0
            Number of subprocesses to use for data loading in PyTorch's DataLoader.

        cache_dir : str, default='./cache'
            Directory for caching the simulated data and associated metadata.

        pop_sizes : Union[List[int], int], default=500
            Number of diploid individuals to sample from each population in the model.
            None = keep all

        num_variants : Union[int, None], default=1000
            Total number of common variants to retain after filtering by minor allele count.

        mac_threshold : int, default=20
            Minor allele count threshold used to filter out rare variants from the genotype matrix.

        sequence_length : float, default=2e7
            Genomic sequence length (in base pairs) used during ancestry simulation. 
            Larger values yield more variants.

        mutation_rate : float, default=1.25e-8
            Per-base mutation rate per generation used for simulating genetic variation.

        recombination_rate : float, default=1e-8
            Per-base recombination rate per generation used for simulating recombination events.

        random_state : int, default=42
            Random seed used for both the ancestry and mutation simulations 
            to ensure reproducibility.

        mode : str, default='split'
            Mode for dataset train/test separation. If 'full', the full dataset is used for both
            training and testing. If 'split', the dataset is partitioned into training and test 
            subsets according to `test_split`.
        """
        
        super().__init__()

        self.batch_size = batch_size
        self.test_split = test_split
        self.num_workers = num_workers
        self.cache_dir = cache_dir

        self.dataset_config = dict(
            cache_dir=cache_dir,
            pop_sizes=pop_sizes,
            num_variants=num_variants,
            mac_threshold=mac_threshold,
            sequence_length=sequence_length,
            mutation_rate=mutation_rate,
            recombination_rate=recombination_rate,
            random_state=random_state
        )

        self.mode = mode
        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # Nothing to download here unless you integrate file caching later
        pass

    def setup(self, stage: str = None):
        if self.mode == "full":
            self.train_dataset = CustomAdmixedModel(**self.dataset_config)
            self.test_dataset = self.train_dataset

        elif self.mode == "split":
            self.dataset = CustomAdmixedModel(**self.dataset_config)
            test_size = int(len(self.dataset) * self.test_split)
            train_size = len(self.dataset) - test_size

            self.train_dataset, self.test_dataset = random_split(
                self.dataset,
                [train_size, test_size],
                generator=torch.Generator().manual_seed(self.dataset_config["random_state"]),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

if __name__ == "__main__":
    data_module = CustomAdmixedModule(
        batch_size=64,
        test_split=0.2,
        pop_size=200,
        num_variants=1000,
        mac_threshold=10,
        mode="split"
    )

    data_module.setup()

    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()

    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    print("Train batch:")
    print("  data shape:", train_batch['data'].shape)
    print("  metadata keys:", train_batch['metadata'].keys())

    print("Test batch:")
    print("  data shape:", test_batch['data'].shape)
    print("  metadata keys:", test_batch['metadata'].keys())