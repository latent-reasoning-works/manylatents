"""
CircleInterpolation synthetic dataset.

Interpolates between two geometric extremes controlled by ``alpha``:

  alpha=0  — noisy circle with nonuniform angular density (= imbalanced circle)
  alpha=1  — tight population centres connected by narrow bridges

Latent variable is an angle theta sampled from a nonuniform density on
[0, 2pi).  The two endpoint geometries are:

  X_circle    = [cos(theta), sin(theta)] + radial noise
  X_popbridge = position interpolated between adjacent sector centres,
                weighted by how close theta is to a sector boundary

  X_alpha = (1 - alpha) * X_circle + alpha * X_popbridge

Rich metadata (theta, bridge_score, X_circle, X_popbridge, …) is stored
as numpy arrays on the dataset object.  The DataLoader-visible ``metadata``
field carries integer sector labels only.

Labels
------
  0 … n_sectors-1  — sector index each point belongs to
"""

import os
from typing import Optional

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from .synthetic_dataset import SyntheticDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _circular_diff(a: np.ndarray, b: float) -> np.ndarray:
    """Signed angular difference a - b wrapped to (-pi, pi]."""
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CircleInterpolation(SyntheticDataset):
    """2D synthetic dataset interpolating between a circle and population bridges.

    Parameters
    ----------
    n_base : int
        Number of points sampled uniformly around the circle before
        adding the oversampled clusters.
    n_sectors : int
        Number of population sectors / clusters.  Sector centres are
        placed at evenly-spaced angles starting at 0.
    oversample_factor : float
        Extra density multiplier near each sector centre.  5.0 means
        each cluster arc contains ~5x the uniform density.
    cluster_arc_deg : float
        Angular half-width (degrees) of the von Mises oversampling kernel.
    alpha : float
        Interpolation coefficient.  0 = pure noisy circle, 1 = pure
        population-bridge geometry.
    circle_noise : float
        Std of Gaussian noise added to the *radius* of X_circle (keeps
        points on the ring, never inside it).
    pop_noise : float
        Isotropic Gaussian noise added to X_popbridge.
    bridge_threshold : float
        bridge_score above which a point is considered a bridge point
        (is_bridge = True).  Range [0, 1]; 0 = always bridge, 1 = never.
    random_state : int
        Reproducibility seed.
    save_figure : bool
        If True, save a 3-panel ground-truth figure on construction.
    save_dir : str
        Directory to write the ground-truth figure.
    """

    def __init__(
        self,
        n_base: int = 1000,
        n_sectors: int = 3,
        oversample_factor: float = 5.0,
        cluster_arc_deg: float = 15.0,
        alpha: float = 0.0,
        circle_noise: float = 0.01,
        pop_noise: float = 0.05,
        bridge_threshold: float = 0.5,
        random_state: int = 42,
        save_figure: bool = False,
        save_dir: str = "outputs",
    ):
        super().__init__()
        rng = np.random.default_rng(random_state)

        self.n_base = n_base
        self.n_sectors = n_sectors
        self.oversample_factor = oversample_factor
        self.cluster_arc_deg = cluster_arc_deg
        self.alpha = float(alpha)
        self.circle_noise = circle_noise
        self.pop_noise = pop_noise
        self.bridge_threshold = bridge_threshold
        self.random_state = random_state
        self.save_dir = save_dir
        self.save_figure = save_figure

        # Sector centres: evenly spaced on [0, 2pi)
        self._sector_mus = np.array(
            [2 * np.pi * k / n_sectors for k in range(n_sectors)]
        )

        # ---- Step 1: sample theta ----------------------------------------
        cluster_arc_rad = np.deg2rad(cluster_arc_deg)
        kappa = 1.0 / (cluster_arc_rad ** 2)
        arc_fraction = (2 * cluster_arc_rad) / (2 * np.pi)
        n_extra = int(round(oversample_factor * arc_fraction * n_base))

        thetas = [rng.uniform(0, 2 * np.pi, size=n_base)]
        sector_src = [np.full(n_base, -1, dtype=np.int64)]  # -1 = uniform

        for k, mu in enumerate(self._sector_mus):
            cluster_theta = rng.vonmises(mu=mu, kappa=kappa, size=n_extra)
            thetas.append(cluster_theta % (2 * np.pi))
            sector_src.append(np.full(n_extra, k, dtype=np.int64))

        theta = np.concatenate(thetas)            # (N,)

        # Assign each theta to its nearest sector centre
        sector_labels = self._assign_sectors(theta)

        # ---- Step 2: X_circle --------------------------------------------
        r = 1.0 + rng.normal(0, circle_noise, size=len(theta))
        r = np.abs(r)
        X_circle = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

        # ---- Step 3: X_popbridge -----------------------------------------
        X_popbridge, bridge_score = self._build_popbridge(theta, sector_labels, rng)

        # ---- Step 4: interpolate -----------------------------------------
        X_alpha = (1 - self.alpha) * X_circle + self.alpha * X_popbridge

        # ---- Step 5: store -----------------------------------------------
        N = len(theta)
        perm = rng.permutation(N)

        self.data = X_alpha[perm].astype(np.float32)
        self.metadata = sector_labels[perm].astype(np.int64)

        # Rich metadata (numpy, not through DataLoader)
        self.theta = theta[perm]
        self.latent_map = np.stack([np.cos(self.theta), np.sin(self.theta)], axis=1)
        self.X_circle = X_circle[perm].astype(np.float32)
        self.X_popbridge = X_popbridge[perm].astype(np.float32)
        self.bridge_score = bridge_score[perm]
        self.is_bridge = self.bridge_score > bridge_threshold

        if self.save_figure:
            self._save_ground_truth_figure()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assign_sectors(self, theta: np.ndarray) -> np.ndarray:
        """Return sector index (0..n_sectors-1) for each angle."""
        diffs = np.stack(
            [np.abs(_circular_diff(theta, mu)) for mu in self._sector_mus],
            axis=1,
        )  # (N, n_sectors)
        return np.argmin(diffs, axis=1).astype(np.int64)

    def _build_popbridge(
        self, theta: np.ndarray, sector_labels: np.ndarray, rng: np.random.Generator
    ):
        """Build X_popbridge and bridge_score for every point."""
        n = len(theta)
        half_sector = np.pi / self.n_sectors

        X = np.zeros((n, 2), dtype=np.float64)
        bridge_score = np.zeros(n, dtype=np.float64)

        for i in range(n):
            s = sector_labels[i]
            mu_s = self._sector_mus[s]
            mu_next = self._sector_mus[(s + 1) % self.n_sectors]

            offset = abs(float(_circular_diff(np.array([theta[i]]), mu_s)[0]))
            bs = min(offset / half_sector, 1.0)
            bridge_score[i] = bs

            P_s = np.array([np.cos(mu_s), np.sin(mu_s)])
            P_next = np.array([np.cos(mu_next), np.sin(mu_next)])
            pos = (1 - bs) * P_s + bs * P_next
            X[i] = pos

        noise = rng.normal(0, self.pop_noise, size=(n, 2))
        return X + noise, bridge_score

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_labels(self, col: Optional[str] = None) -> np.ndarray:
        """Return integer labels for colouring.

        Parameters
        ----------
        col : str or None
            ``None`` / ``'sector'``: sector index (default, used by PlotEmbeddings)
            ``'theta'``: theta quantised into 64 equal-width bins
            ``'bridge'``: binary is_bridge flag
            ``'bridge_score'``: bridge_score quantised into 32 bins
        """
        if col is None or col == "sector":
            return self.metadata
        if col == "theta":
            bins = np.linspace(0, 2 * np.pi, 65)
            return (np.digitize(self.theta, bins) - 1).astype(np.int64)
        if col == "bridge":
            return self.is_bridge.astype(np.int64)
        if col == "bridge_score":
            bins = np.linspace(0, 1, 33)
            return (np.digitize(self.bridge_score, bins) - 1).astype(np.int64)
        # Unknown col — fall back to sector labels
        return self.metadata

    def get_gt_dists(self) -> np.ndarray:
        """Pairwise circular geodesic distances from theta.

        Returns
        -------
        np.ndarray, shape (N, N)
        """
        diff = self.theta[:, None] - self.theta[None, :]
        return np.minimum(np.abs(diff), 2 * np.pi - np.abs(diff))

    def get_colormap_info(self):
        from manylatents.callbacks.embedding.base import ColormapInfo

        label_names = {k: f"sector_{k}" for k in range(self.n_sectors)}
        return ColormapInfo(
            cmap="tab10",
            label_names=label_names,
            is_categorical=True,
        )

    def _save_ground_truth_figure(self):
        """Save 3-panel ground-truth scatter to save_dir."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(self.save_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"CircleInterpolation ground truth  (n_sectors={self.n_sectors}, alpha={self.alpha})",
            fontsize=12,
        )

        # Panel 1: X_circle coloured by theta
        sc0 = axes[0].scatter(
            self.X_circle[:, 0], self.X_circle[:, 1],
            c=self.theta, cmap="hsv", s=4, alpha=0.6,
        )
        axes[0].set_title("X_circle  (coloured by theta)")
        plt.colorbar(sc0, ax=axes[0], label="theta (rad)")

        # Panel 2: X_alpha coloured by sector
        sc1 = axes[1].scatter(
            self.data[:, 0], self.data[:, 1],
            c=self.metadata, cmap="tab10",
            vmin=-0.5, vmax=self.n_sectors - 0.5,
            s=4, alpha=0.6,
        )
        axes[1].set_title(f"X_alpha (alpha={self.alpha})  (coloured by sector)")
        plt.colorbar(sc1, ax=axes[1], label="sector")

        # Panel 3: X_popbridge coloured by bridge_score
        sc2 = axes[2].scatter(
            self.X_popbridge[:, 0], self.X_popbridge[:, 1],
            c=self.bridge_score, cmap="plasma", vmin=0, vmax=1,
            s=4, alpha=0.6,
        )
        axes[2].set_title("X_popbridge  (coloured by bridge_score)")
        plt.colorbar(sc2, ax=axes[2], label="bridge_score")

        for ax in axes:
            ax.set_aspect("equal")

        path = os.path.join(self.save_dir, "circle_interpolation_ground_truth.png")
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class CircleInterpolationDataModule(LightningDataModule):
    """LightningDataModule wrapping :class:`CircleInterpolation`."""

    def __init__(
        self,
        batch_size: int = 128,
        num_workers: int = 0,
        shuffle_traindata: bool = False,
        test_split: float = 0.2,
        mode: str = "full",
        n_base: int = 1000,
        n_sectors: int = 3,
        oversample_factor: float = 5.0,
        cluster_arc_deg: float = 15.0,
        alpha: float = 0.0,
        circle_noise: float = 0.01,
        pop_noise: float = 0.05,
        bridge_threshold: float = 0.5,
        random_state: int = 42,
        save_figure: bool = False,
        save_dir: str = "outputs",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_traindata = shuffle_traindata
        self.test_split = test_split
        self.mode = mode

        self.n_base = n_base
        self.n_sectors = n_sectors
        self.oversample_factor = oversample_factor
        self.cluster_arc_deg = cluster_arc_deg
        self.alpha = alpha
        self.circle_noise = circle_noise
        self.pop_noise = pop_noise
        self.bridge_threshold = bridge_threshold
        self.random_state = random_state
        self.save_figure = save_figure
        self.save_dir = save_dir

        self.train_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = None):
        ds = CircleInterpolation(
            n_base=self.n_base,
            n_sectors=self.n_sectors,
            oversample_factor=self.oversample_factor,
            cluster_arc_deg=self.cluster_arc_deg,
            alpha=self.alpha,
            circle_noise=self.circle_noise,
            pop_noise=self.pop_noise,
            bridge_threshold=self.bridge_threshold,
            random_state=self.random_state,
            save_figure=self.save_figure,
            save_dir=self.save_dir,
        )

        if self.mode == "full":
            self.train_dataset = ds
            self.test_dataset = ds
        elif self.mode == "split":
            test_size = int(len(ds) * self.test_split)
            train_size = len(ds) - test_size
            self.train_dataset, self.test_dataset = random_split(
                ds,
                [train_size, test_size],
                generator=torch.Generator().manual_seed(self.random_state),
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode!r}. Use 'full' or 'split'.")

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
