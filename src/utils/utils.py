import os
import pickle

import numpy as np
import pandas as pd

from utils import helpers
from utils import metrics

# Helper function to save/load objects

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_data(admixtures_k, data_dir, admixture_dir, base_path, fname):
    # Step -1: Load data
    merged_metadata, relatedness, genotypes_array, mapping_info = helpers.load_data(base_path, fname)

    # Step 0: Pre-process data
    normalized_matrix, overlap_counts = helpers.preprocess_data_matrix(genotypes_array)

    # Fit PCA model on unrelated samples
    filters = ["filter_pca_outlier", "hard_filtered", "filter_contaminated"]
    _filtered_indices = merged_metadata[merged_metadata[filters].any(axis=1)].index
    filtered_indices = ~merged_metadata.index.isin(_filtered_indices)
    related_indices = ~merged_metadata['filter_king_related'].values

    to_fit_on = related_indices & filtered_indices
    to_transform_on = (~related_indices) & filtered_indices

    pca_emb, _ = helpers.compute_pca_from_hail(
        os.path.join(base_path, 'pca_scores_hailcomputed.csv'),
        merged_metadata,
        50
    )

    admixture_ratios_list = []
    prefix = 'global'
    for n_comps in admixtures_k:
        fname = f"{prefix}.{n_comps}_metadata.tsv"
        admix_ratios = pd.read_csv(os.path.join(admixture_dir, fname), sep='\t', header=None)

        admixture_ratios_nonzero = admix_ratios.loc[:, 1:n_comps].values
        admixture_ratios = np.zeros((pca_emb.shape[0], admixture_ratios_nonzero.shape[1]))

        index = to_fit_on | to_transform_on
        admixture_ratios[index] = admixture_ratios_nonzero
        admixture_ratios_list.append(admixture_ratios)

    return pca_emb, merged_metadata, to_fit_on, to_transform_on, admixture_ratios_list, mapping_info[1]

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