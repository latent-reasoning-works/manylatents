import os
import h5py
import numpy as np
import pandas as pd
from typing import Tuple, List
import mappings

def load_data_1000G(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load genetic data from HDF5 file format.

    Args:
        file_path (str): Path to the .hdf5 file.

    Returns:
        Tuple[np.ndarray, ...]: Tuple containing inputs, class labels, sample IDs, and SNP names.
    """
    with h5py.File(file_path, 'r') as hf:
        inputs = hf['inputs'][:]
        class_labels = hf['class_labels'][:]
        samples = hf['samples'][:]
        snp_names = hf['snp_names'][:]
        class_label_names = hf['class_label_names'][:]

    return inputs, class_labels, samples, snp_names, class_label_names

def preprocess_labels_1000G(class_labels: np.ndarray, class_label_names: np.ndarray) -> List[str]:
    """Preprocess class labels to readable format.

    Args:
        class_labels (np.ndarray): Array of class label indices.
        class_label_names (np.ndarray): Array of class label names.

    Returns:
        List[str]: List of label names corresponding to class indices.
    """
    
    # fine grained labels
    label_names_fine = [str(class_label_names[label])[2:-1] for label in class_labels]
    
    # coarse granined labels
    label_names_coarse = np.zeros_like(label_names_fine)
    for label in mappings.super_pops_1000G:
        index = pd.DataFrame(label_names_fine).isin(mappings.super_pops_1000G[label]).values.flatten()
        label_names_coarse[index] = label
    
    return label_names_fine, label_names_coarse

def load_metadata(file_path: str) -> pd.DataFrame:
    """Load metadata from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Metadata as a DataFrame.
    """
    return pd.read_csv(file_path, sep='\t')


def preprocess_labels_sarscov2(metadata):
    # labels fine
    metadata['WHO'] = metadata['WHO'].replace(mappings.replace_dict)  # replace_dict should be defined/imported
    metadata['WHO'] = np.where(metadata['WHO'].isin(mappings.allowed_labels), 
                                   metadata['WHO'], 
                                   'Other')
    
    #labels coarse
    #labels = merged[~((merged['WHO'] == 'Other') | (merged['WHO'] == 'Unassigned'))]['WHO']
    labels_coarse = metadata['WHO'].apply(lambda x: 'Omicron' if x.startswith('Omicron') else x)

    return metadata['WHO'], labels_coarse
