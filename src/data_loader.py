import os
import h5py
import numpy as np
import pandas as pd
from typing import Tuple, List
import mappings


def load_data_HGDP(data_path: str, 
                   metadata_path: str,
                   unrelated_sampleid_1000G_path: str) -> Tuple[np.ndarray, 
                                                         np.ndarray, 
                                                         np.ndarray,
                                                         np.ndarray, 
                                                         np.ndarray, 
                                                         pd.DataFrame]:

    with h5py.File(data_path, 'r') as hf:
        #model_attrs = hf['gradients'][:]
        #print('loaded gradient of fc1 w.r.t. input from {}'.format(attr_fc1_saliency_name))
        inputs = hf['inputs'][:]
        class_label_names = np.char.decode(hf['class_label_names'][:])
        class_labels = hf['class_labels'][:]
        samples = hf['samples'][:]
        snp_names = np.char.decode(hf['snp_names'][:])

    # Remove first sample because it is some kind of dummy values
    sample = np.char.decode(samples)[1:]
    inputs = inputs[1:]

    # Load metadata from file
    metadata_labels = pd.read_csv(metadata_path, sep='\t')

    # Create Superpop/pop labels
    population_info = metadata_labels.apply(lambda row: row.Population_Genetic_region.split('_'), axis='columns', result_type='expand')
    metadata_labels = pd.concat([metadata_labels['sample'], population_info], axis='columns')[1:]
    metadata_labels = metadata_labels.rename(columns={0: 'Population', 1: 'Superpopulation'})

    # identify 1000G samples (using column: db)
    db_labels = pd.Categorical(['HGDP']*len(metadata_labels['Population']), categories=['1000G', 'HGDP'])
    db_labels[metadata_labels['Population'].isin(mappings.label_order_1000G_fine + ['CEU', 'GBR', 'STU', 'ITU'])] = '1000G'
    metadata_labels['db'] = db_labels
    metadata_labels = metadata_labels.set_index('sample').loc[sample.tolist()]

    # Load unrelated individuals counts
    unrelated = pd.read_csv(unrelated_sampleid_1000G_path, header=None)
    unrelated = unrelated[0].tolist()

    thG_data = metadata_labels[metadata_labels['db']=='1000G']
    # This is a set of unrelated indivuals in unrelated_sampleid_1000G_path 
    unrelated_bool_idx = thG_data.index.isin(unrelated)
    unrelated_idx = thG_data.index[unrelated_bool_idx]
    
    print('Removed {} related individuals'.format((~unrelated_bool_idx).sum()))
    
    # Note: 'NA12546', 'NA12830', 'NA18874' appear in unrelated but not in 1000G data
    full_tokeep = metadata_labels[metadata_labels['db']=='HGDP'].index.tolist() + unrelated_idx.tolist()
    
    sample_df = pd.DataFrame({'sample': sample})
    tmp_df = pd.merge(sample_df, pd.DataFrame(inputs), left_index=True, right_index=True)
    tmp_df = tmp_df.set_index('sample')

    inputs = tmp_df.loc[full_tokeep].values
    metadata_labels = metadata_labels.loc[full_tokeep]
    sample = full_tokeep
    
    return inputs, class_labels, sample, snp_names, class_label_names, metadata_labels


def load_data_MHI(data_path: str, 
                  metadata_path: str) -> Tuple[np.ndarray, 
                                               np.ndarray, 
                                               np.ndarray,
                                               np.ndarray, 
                                               np.ndarray, 
                                               pd.DataFrame]:

    with h5py.File(data_path, 'r') as hf:
        inputs = hf['inputs'][:]
        class_label_names = hf['class_label_names'][:]
        class_labels = hf['class_labels'][:]
        samples = hf['samples'][:]
        snp_names = hf['snp_names'][:]

    samples = np.char.decode(samples)

    metadata_labels = pd.read_csv(metadata_path, sep='\t')

    # Set the 'ID' column as the index
    metadata_labels_indexed = metadata_labels.set_index('ID')

    # remove labels that dont appear in df
    intersection = metadata_labels_indexed.index.isin(samples.tolist()) 
    metadata_labels_indexed = metadata_labels_indexed[intersection]

    # remove samples in df that dont appear in labels
    intersection2 = pd.DataFrame(samples)[0].isin(metadata_labels_indexed.index)
    samples = samples[intersection2]
    inputs = inputs[intersection2]

    # Reorder the DataFrame based on id_reorder
    metadata_labels_reordered = metadata_labels_indexed.loc[samples.tolist()]

    return inputs, class_labels, samples, snp_names, class_label_names, metadata_labels_reordered

def load_data_1000G(file_path: str) -> Tuple[np.ndarray, 
                                             np.ndarray, 
                                             np.ndarray, 
                                             np.ndarray, 
                                             np.ndarray]:
    """Load genetic data from HDF5 file format.

    Args:
        file_path (str): Path to the .hdf5 file.

    Returns:
        Tuple[np.ndarray, ...]: Tuple containing inputs, class labels, sample IDs, and SNP names.
    """
    with h5py.File(file_path, 'r') as hf:
        inputs = hf['inputs'][:]
        class_labels = hf['class_labels'][:]
        samples = np.char.decode(hf['samples'][:])
        snp_names = np.char.decode(hf['snp_names'][:])
        class_label_names = np.char.decode(hf['class_label_names'][:])

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
    label_names_fine = [str(class_label_names[label]) for label in class_labels]
    
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
