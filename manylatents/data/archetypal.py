import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from typing import List, Optional
from .synthetic_dataset import Archetypal


class ArchetypalDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for the Archetypal dataset.

    Samples from a simplex (optionally projected onto a sphere) with
    Dirichlet-distributed points that naturally concentrate near corners
    and edges, producing archetypal data.
    """

    def __init__(
        self,
        batch_size: int = 128,
        test_split: float = 0.2,
        num_workers: int = 0,
        shuffle_traindata: bool = True,
        n_components: int = 3,
        simplex_radius: float = 1.0,
        n_obs: int = 5000,
        concentration: float = 0.3,
        noise: float = 0.0,
        random_state: int = 42,
        output_dims: int = 0,
        mode: str = "full",
        use_gap: bool = False,
        n_gaps: int = 0,
        project_to_sphere: bool = True,
        vertex_weights: Optional[List[float]] = None,
        save_figure: bool = False,
        save_dir: str = "outputs",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.shuffle_traindata = shuffle_traindata
        self.test_split = test_split
        self.num_workers = num_workers

        self.n_components = n_components
        self.simplex_radius = simplex_radius
        self.n_obs = n_obs
        self.concentration = concentration
        self.noise = noise
        self.random_state = random_state
        self.output_dims = output_dims

        self.mode = mode
        self.use_gap = use_gap
        self.n_gaps = n_gaps
        self.project_to_sphere = project_to_sphere
        self.vertex_weights = vertex_weights
        self.save_figure = save_figure
        self.save_dir = save_dir

        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        pass

    def _make_dataset(self):
        return Archetypal(
            n_components=self.n_components,
            simplex_radius=self.simplex_radius,
            n_obs=self.n_obs,
            concentration=self.concentration,
            noise=self.noise,
            random_state=self.random_state,
            output_dims=self.output_dims,
            use_gap=self.use_gap,
            n_gaps=self.n_gaps,
            project_to_sphere=self.project_to_sphere,
            vertex_weights=self.vertex_weights,
            save_figure=self.save_figure,
            save_dir=self.save_dir,
        )

    def setup(self, stage: str = None):
        if self.mode == "full":
            self.train_dataset = self._make_dataset()
            self.test_dataset = self.train_dataset

        elif self.mode == "split":
            self.dataset = self._make_dataset()

            test_size = int(len(self.dataset) * self.test_split)
            train_size = len(self.dataset) - test_size

            self.train_dataset, self.test_dataset = random_split(
                self.dataset,
                [train_size, test_size],
                generator=torch.Generator().manual_seed(self.random_state),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_traindata,
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
