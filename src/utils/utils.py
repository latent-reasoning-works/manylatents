import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Helper function to save/load objects

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def check_or_make_dirs(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
 
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
    Prepare directories based on the configuration.

    Args:
        cfg (DictConfig): Configuration object.
    """
    # Hydra's working directory is set to the run's output folder for specific outputs
    ## TODO: leverage specific hydra outputs
    run_dir = Path.cwd()

    create_directory(cfg.paths.ckpt_dir)

    if cfg.project.plotting:
        plot_dir = run_dir / "plots"
        create_directory(plot_dir)

    # General cache directory outside Hydra run-specific folders
    cache_dir = Path(cfg.paths.cache_dir)
    geodesic_dir = cache_dir / "geodesic"
    laplacian_dir = cache_dir / "laplacian"

    if cfg.project.caching:
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

def detect_separator(file_path: str, sample_size: int = 1024) -> Optional[str]:
    """
    Detects the delimiter of a file using csv.Sniffer.

    Args:
        file_path (str): Path to the file.
        sample_size (int): Number of bytes to read for detection.

    Returns:
        Optional[str]: Detected delimiter or None if detection fails.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = f.read(sample_size)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            delimiter = dialect.delimiter
            logging.info(f"Detected delimiter '{delimiter}' for file '{file_path}'.")
            return delimiter
    except Exception as e:
        logging.warning(f"Could not detect delimiter for file '{file_path}': {e}")
        return None
