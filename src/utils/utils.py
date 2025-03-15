import csv
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import rich
import rich.logging
import torch

import wandb

logger = logging.getLogger(__name__)

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

def save_embeddings(embeddings, path, format='npy', metadata=None):
    """
    Saves embeddings in the specified format.

    Args:
        embeddings (np.ndarray): The computed embeddings.
        path (str): File path for saving.
        format (str): One of ['npy', 'csv', 'pt', 'h5'].
        metadata (dict, optional): Extra metadata (e.g., labels).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    if format == 'npy':
        np.save(path, embeddings)
    elif format == 'csv':
        df = pd.DataFrame(embeddings, columns=[f"Component_{i+1}" for i in range(embeddings.shape[1])])
        if metadata is not None and metadata.get("labels") is not None:
            df["label"] = metadata["labels"]
        df.to_csv(path, index=False)
    elif format == 'pt':
        torch.save(embeddings, path)
    elif format == 'h5':
        with h5py.File(path, 'w') as f:
            f.create_dataset('embeddings', data=embeddings)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
def setup_logging(debug: bool = False):
    """
    Configures logging dynamically using Hydra's built-in system.
    
    Args:
        debug (bool): If True, enables local-only logging without WandB.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Remove existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Logging (Rich Formatting)
    console_handler = rich.logging.RichHandler(
        markup=True, rich_tracebacks=True, tracebacks_width=100, tracebacks_show_locals=False
    )
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)

    logger.addHandler(console_handler)

    logger.info("Logging system initialized successfully.")
    
def aggregate_metrics(
    dr_metrics: dict = None,
    dr_error: float = None,
    model_metrics: dict = None,
    model_error: float = None,
    callback_outputs: list = None
) -> dict:
    """
    Aggregates metrics from DR evaluation, network evaluation, and callbacks,
    and logs them to wandb if a run is active.
    
    Returns:
        A dictionary containing all aggregated metrics.
    """
    aggregated_metrics = {}

    # Merge DR metrics.
    if dr_metrics:
        aggregated_metrics.update(dr_metrics)
    if dr_error is not None:
        aggregated_metrics["DR_Error"] = dr_error

    # Merge network (model) metrics.
    if model_metrics:
        aggregated_metrics.update(model_metrics)
    if model_error is not None:
        aggregated_metrics["Model_Error"] = model_error

    # Process any callback outputs.
    if callback_outputs:
        for name, output in callback_outputs:
            # If the callback returns a dictionary of metrics, prefix keys with the callback name.
            if isinstance(output, dict):
                prefixed = {f"{name}_{k}": v for k, v in output.items()}
                aggregated_metrics.update(prefixed)
            # If output is an image path, log it as a wandb.Image.
            elif isinstance(output, str) and output.endswith(".png"):
                aggregated_metrics[f"{name}_plot"] = wandb.Image(output)
            # Otherwise, log the string directly.
            elif isinstance(output, str):
                aggregated_metrics[name] = output

    return aggregated_metrics
