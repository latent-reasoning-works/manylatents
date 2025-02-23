import os
import copy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from scipy.sparse.csgraph import connected_components

import networkx as nx
from matplotlib.patches import Patch
from typing import List, Dict, Tuple, Union

def gen_dla_with_disconnectivity(
    n_dim: int = 100,
    n_branch: int = 20,
    branch_lengths: Union[List[int], int, None] = None,  # Fixing type hint
    rand_multiplier: float = 2,
    gap_multiplier: float = 0,
    seed: int = 37,
    sigma: float = 4,
    disconnect_branches: List[int] = [5, 15],  # Fixing list typing
    sampling_density_factors: Union[Dict[int, float], None] = None  # Fixing dict typing
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
    """
    Generates a Diffusion-Limited Aggregation (DLA) tree with optional branch disconnections.

    Parameters:
    - n_dim (int): Number of dimensions for each point in the tree.
    - n_branch (int): Number of branches in the tree.
    - branch_lengths (list[int] | int | None): Length of each branch. If int, all branches have the same length.
    - rand_multiplier (float): Scaling factor for random movements.
    - gap_multiplier (float): Scaling factor for disconnection jumps.
    - seed (int): Random seed for reproducibility.
    - sigma (float): Standard deviation of Gaussian noise added to the dataset.
    - disconnect_branches (list[int]): Indices of branches to disconnect.
    - sampling_density_factors (dict[int, float] | None): Dictionary specifying a density reduction factor per branch.
        - Keys are branch indices.
        - Values are float factors (e.g., 0.5 means keeping 50% of points in that branch).

    Returns:
    - tuple[np.ndarray, np.ndarray, np.ndarray]:
        - M (np.ndarray): Noisy dataset with possible disconnections.
        - M_gt (np.ndarray): Ground truth dataset without disconnections.
        - C (np.ndarray): Labels indicating branch membership of each point.
    """
    np.random.seed(seed)

    # Default branch lengths if none are provided
    if branch_lengths is None:
        branch_lengths = [100] * n_branch
    elif isinstance(branch_lengths, int):
        branch_lengths = [branch_lengths] * n_branch
    elif len(branch_lengths) != n_branch:
        raise ValueError("The length of 'branch_lengths' must match 'n_branch'.")

    # Initialize the first branch
    M_gt = np.cumsum(-1 + rand_multiplier * np.random.rand(branch_lengths[0], n_dim), axis=0)
    M = M_gt.copy()  # Start with ground truth dataset
    branch_start_indices = [0]  # Keep track of where each branch starts

    for i in range(1, n_branch):
        ind = np.random.randint(branch_start_indices[i - 1], branch_start_indices[i - 1] + branch_lengths[i - 1])

        # Create the ground truth branch first
        new_branch_gt = np.cumsum(
            -1 + rand_multiplier * np.random.rand(branch_lengths[i], n_dim), axis=0
        )
        new_branch_gt += M_gt[ind]

        # Create the potentially disconnected branch
        new_branch = new_branch_gt.copy()
        if i in disconnect_branches:
            jump = np.random.normal(gap_multiplier, 0.1, n_dim)  # Jump offset
            new_branch += jump  # Apply the jump to all points in the branch

            # Check if the jump places the branch too close to another branch
            distances = np.linalg.norm(M - new_branch[0], axis=1)
            if np.min(distances) < rand_multiplier:
                raise ValueError(f"Jump for branch {i} is too close to another branch. Adjust gap_multiplier.")

        M_gt = np.concatenate([M_gt, new_branch_gt])
        M = np.concatenate([M, new_branch])
        branch_start_indices.append(M.shape[0] - branch_lengths[i])

    # Reduce sampling density for certain branches
    if sampling_density_factors:
        mask = np.ones(M.shape[0], dtype=bool)
        for branch_idx, factor in sampling_density_factors.items():
            start_idx = branch_start_indices[branch_idx]
            end_idx = start_idx + branch_lengths[branch_idx]
            branch_points = np.arange(start_idx, end_idx)
            keep_points = np.random.choice(branch_points, int(len(branch_points) * factor), replace=False)
            mask[branch_points] = False  # Remove points
            mask[keep_points] = True  # Retain selected points
        M = M[mask]
        M_gt = M_gt[mask]  # Apply the same mask to the ground truth

    # Add noise
    noise = np.random.normal(0, sigma, M.shape)
    M = M + noise
    M_gt = M_gt + noise

    # Update group labels for visualization
    C = np.array(
        [
            i for branch_idx, branch_len in enumerate(branch_lengths)
            for i in [branch_idx] * branch_len
        ]
    )
    C = C[mask] if sampling_density_factors else C

    # Return both datasets if disconnection is applied
    if disconnect_branches:
        return M, M_gt, C
    else:
        return M, C


def make_sim_data(n_dim=200, n_branch=6, branch_lengths=120, 
                  rand_multiplier=2, seed=37, sigma=5, disconnect_branches=[0], gap_multiplier=5,
                  sampling_density_factors=None):
    tree, tree_gt, branches = gen_dla_with_disconnectivity(n_dim=n_dim, 
                                                  n_branch=n_branch, branch_lengths=branch_lengths, 
                                                  rand_multiplier=rand_multiplier, 
                                                  seed=seed, sigma=sigma, gap_multiplier=gap_multiplier,
                                                  disconnect_branches=disconnect_branches,
                                                  sampling_density_factors=sampling_density_factors)

    return tree, tree_gt, branches