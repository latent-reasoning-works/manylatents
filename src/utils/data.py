import csv
import logging
import os
from typing import Callable, Optional

import numpy as np
import pandas as pd
from pyplink import PyPlink
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as svstack
from tqdm import tqdm

from .mappings import make_palette_label_order_HGDP

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
    except csv.Error as e:
        logging.warning(f"CSV parsing error while detecting delimiter for '{file_path}': {e}")
        return None
    except UnicodeDecodeError as e:
        logging.warning(f"Encoding error while reading '{file_path}': {e}")
        return None
    except Exception as e:
        logging.warning(f"Unexpected error while detecting delimiter for '{file_path}': {e}")
        return None

def load_metadata(
    file_path: str,
    required_columns: Optional[list] = None,
    additional_processing: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
) -> pd.DataFrame:
    """
    Loads metadata from a file, handling delimiter detection and basic validation.

    Args:
        file_path (str): Path to the metadata file.
        required_columns (Optional[list]): List of columns that must be present.
        additional_processing (Optional[Callable[[pd.DataFrame], pd.DataFrame]]): 
            A function to perform additional processing on the DataFrame.

    Returns:
        pd.DataFrame: Loaded and processed metadata.
    """
    logging.info(f"Loading metadata from: {file_path}")

    # Detect delimiter
    delimiter = detect_separator(file_path)

    # Fallback to tab if detection fails
    if delimiter is None:
        logging.warning(f"Falling back to tab separator for file: {file_path}")
        delimiter = '\t'

    try:
        metadata = pd.read_csv(file_path, sep=delimiter, engine='python')
        logging.info(f"Successfully loaded metadata with delimiter '{delimiter}'.")
    except pd.errors.ParserError as e:
        logging.error(f"ParserError while reading {file_path}: {e}")
        raise
    except FileNotFoundError:
        logging.error(f"Metadata file not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error while reading {file_path}: {e}")
        raise

    # Validate required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in metadata.columns]
        if missing_columns:
            logging.error(f"Missing required columns {missing_columns} in metadata file: {file_path}")
            raise KeyError(f"Missing required columns {missing_columns} in metadata file: {file_path}")

    # Perform additional processing if provided
    if additional_processing:
        metadata = additional_processing(metadata)

    return metadata
