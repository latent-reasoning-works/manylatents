import os
import h5py
import numpy as np
import pandas as pd
from typing import Tuple, List
import mappings
from pyplink import PyPlink
import tqdm
from mappings import make_palette_label_order_HGDP

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

def load_data_HGDP(exp_path):
    # Load HGDP data
    try:
        genotypes_array = np.load(exp_path + 'V4_raw_genotypes.npy')
    except:
        fname = 'gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.match1000G_GSAs_dietnet'
        data_path = os.path.join(exp_path, fname)
        pedfile = PyPlink(data_path)
        genotypes_array = np.zeros([pedfile.get_nb_samples(), pedfile.get_nb_markers()], dtype=np.int8)

        for i, (marker_id, genotypes) in tqdm.tqdm(enumerate(pedfile)):
            genotypes_array[:,i] = genotypes

        np.save(exp_path + 'V4_raw_genotypes.npy', genotypes_array)

    labels = pd.read_csv(os.path.join(exp_path, 'gnomad.genomes.v3.1.2.hgdp_1kg_subset_sample_meta.reduced.tsv'), sep='\t')

    genotypes_array = genotypes_array[1:] # remove first row
    labels = labels[1:] # remove first row

    # Load filter data
    filter_info = pd.read_csv(os.path.join(exp_path, '4.3/gnomad_derived_metadata_with_filtered_sampleids.csv'), sep=',', index_col=1)

    merged_metadata = labels.set_index('sample').merge(filter_info, left_index=True, right_index=True)

    # load relatedness
    relatedness = pd.read_csv(os.path.join(exp_path, '4.3/HGDP+1KGP_MattEstimated_king_relatedness_matrix.csv'), sep=',', index_col=0)
    #cols_to_filter = relatedness.index[(~merged_metadata.loc[relatedness.index]['filter_king_related']).values].values
    #relatedness_none_related = relatedness[(~merged_metadata.loc[relatedness.index]['filter_king_related']).values][cols_to_filter]
    
    pop_palette_hgdp_coarse, pop_palette_hgdp_fine, label_order_hgdp_coarse, label_order_hgdp_fine = make_palette_label_order_HGDP(merged_metadata['Population'], merged_metadata['Genetic_region'])
    
    return merged_metadata, relatedness, genotypes_array, (pop_palette_hgdp_coarse, 
                                                           pop_palette_hgdp_fine, 
                                                           label_order_hgdp_coarse, 
                                                           label_order_hgdp_fine)
    
