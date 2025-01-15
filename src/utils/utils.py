import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Helper function to save/load objects

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def create_directory(dir_path, condition=True):
    """
    Create a directory if the condition is True.

    Args:
        dir_path (str or Path): Path to the directory.
        condition (bool): Whether to create the directory.
    """
    if condition:
        os.makedirs(dir_path, exist_ok=True)


def prepare_directories(cfg):
    """
    Prepare directories based on the configuration and Hydra's run directory.

    Args:
        cfg (DictConfig): Configuration object.
    """
    # Get Hydra's working directory
    base_dir = Path.cwd()

    # Fixed directory
    ckpt_dir = base_dir / "ckpt"
    create_directory(ckpt_dir)

    # Conditional directories
    if cfg.project.plotting:
        plot_dir = base_dir / "plots"
        create_directory(plot_dir)

    if cfg.project.caching:
        cache_dir = base_dir / "cache"
        geodesic_dir = cache_dir / "geodesic"
        laplacian_dir = cache_dir / "laplacian"

        create_directory(cache_dir)
        create_directory(geodesic_dir)
        create_directory(laplacian_dir)
        
# Convert results to DataFrame
def create_results_dataframe(results):
    # Get all possible keys
    all_keys = set(key for result in results for key in result.keys())

    # Ensure all dictionaries have the same keys
    normalized_results = [
        {key: result.get(key, np.nan) for key in all_keys} for result in results
    ]

    # Convert to DataFrame
    return pd.DataFrame(normalized_results)