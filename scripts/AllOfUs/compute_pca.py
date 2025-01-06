import os
import numpy as np
import tqdm
import numpy.ma as ma
import h5py
from sklearn.decomposition import PCA
from pyplink import PyPlink
import argparse

def main(plink_file_root, plink_file_name, emb_folder, subset='1KGP', pca_components=50, chunk_size=5000):
    # Paths
    emb_folder
    prefix = plink_file_name.split('.')[0]  # Infer the prefix from plink file name, e.g., 'extractedChrAllPruned'
    pca_path = os.path.join(emb_folder, f'{prefix}_pca_proj_{subset}.h5')

    # Create the embedding folder if it doesn't exist
    os.makedirs(emb_folder, exist_ok=True)

    # Check if PCA has been computed
    if os.path.exists(pca_path):
        print(f"PCA embeddings for {subset} already exist at {pca_path}. Skipping PCA.")
        return

    # Load plink data
    pedfile = PyPlink(os.path.join(plink_file_root, prefix))
    all_samples = pedfile.get_fam()

    # Get sample indexes for the specified subset (for PCA fitting)
    sample_idxs = get_sample_indexes(all_samples, subset)

    # Load or compute the genotype array for the specified subset
    genotype_npy_path = os.path.join(emb_folder, f'{prefix}_{subset}_genotypes.npy')
    genotypes_array = load_or_compute_genotypes(pedfile, sample_idxs, genotype_npy_path)

    # Preprocess the genotype array for the subset
    colmeans_npy_path = os.path.join(emb_folder, f'{prefix}_{subset}_genotypes_colmeans.npy')
    scaled_genotypes_array, colmeans = preprocess_genotypes(genotypes_array, colmeans_npy_path)

    # Run PCA on the subset (filtered samples)
    pca_obj = PCA(n_components=pca_components)
    print(f"Fitting PCA on the {subset} subset...")
    pca_obj.fit(scaled_genotypes_array)

    # Transform the full dataset in chunks
    path_to_genotypes = os.path.join(emb_folder, f'{prefix}_raw_genotypes.npy')
    full_genotypes_array = load_or_compute_full_genotypes(pedfile, path_to_genotypes)

    # Transform the full dataset in chunks and save the PCA embeddings
    transform_full_dataset_in_chunks(pca_obj, full_genotypes_array, colmeans, pca_path, chunk_size)

def get_sample_indexes(all_samples, subset):
    """Get the sample indexes for a specified subset (default is '1KGP')."""
    if subset == '1KGP':
        IKGP_label_order = ['YRI', 'ESN', 'GWD', 'LWK', 'MSL', 'ACB', 'ASW',
                            'IBS',  'CEU', 'GBR', 'TSI', 'FIN',
                            'PJL', 'BEB', 'GIH', 'STU', 'ITU',
                            'CHB', 'CHS', 'CDX', 'KHV', 'JPT',
                            'MXL', 'CLM', 'PEL', 'PUR']
    else:
        raise ValueError(f"Subset '{subset}' is not recognized.")

    sample_idxs = [np.where(all_samples['iid'].str.startswith(k))[0] for k in IKGP_label_order]
    sample_idxs = np.concatenate(sample_idxs)
    return sample_idxs

def load_or_compute_genotypes(pedfile, sample_idxs, path):
    try:
        genotypes_array = np.load(path)
        print(f"Loaded genotypes for subset from {path}")
    except:
        print(f"Error loading genotypes for subset from {path}. Recomputing...")
        genotypes_array = np.zeros([sample_idxs.shape[0], pedfile.get_nb_markers()], dtype=np.int8)

        for i, (marker_id, genotypes) in tqdm.tqdm(enumerate(pedfile)):
            genotypes_array[:, i] = genotypes[sample_idxs]

        np.save(path, genotypes_array)
    return genotypes_array

def preprocess_genotypes(genotypes_array, colmeans_path):
    genotypes_array_float = genotypes_array.astype(float)
    genotypes_array_float[genotypes_array_float == -1] = np.nan  # replace -1 with nan

    # Load or compute the column means
    try:
        colmeans = np.load(colmeans_path)
        print(f"Loaded column means from {colmeans_path}")
    except:
        colmeans = np.nanmean(genotypes_array_float, axis=0)
        np.save(colmeans_path, colmeans)

    # Scale the data by subtracting column means (center the data)
    scaled_genotypes_array = np.where(np.isnan(genotypes_array_float),
                                      np.nanmean(genotypes_array_float, axis=0),
                                      genotypes_array_float)
    scaled_genotypes_array -= colmeans
    return scaled_genotypes_array, colmeans

def load_or_compute_full_genotypes(pedfile, path):
    try:
        genotypes_array = np.load(path)
        print(f"Loaded full genotypes from {path}")
    except:
        print(f"Error loading full genotypes from {path}. Recomputing...")
        genotypes_array = np.zeros([pedfile.get_nb_samples(), pedfile.get_nb_markers()], dtype=np.int8)

        for i, (marker_id, genotypes) in tqdm.tqdm(enumerate(pedfile)):
            genotypes_array[:, i] = genotypes

        np.save(path, genotypes_array)
    return genotypes_array

def transform_full_dataset_in_chunks(pca_obj, full_genotypes_array, colmeans, pca_path, chunk_size):
    num_samples = full_genotypes_array.shape[0]
    num_components = pca_obj.n_components_

    # Open HDF5 file to store PCA projections
    with h5py.File(pca_path, 'w') as h5f:
        transformed_dataset = h5f.create_dataset('transformed_data', 
                                                 shape=(num_samples, num_components), 
                                                 dtype='float32')

        # Process the dataset in chunks
        for i in tqdm.tqdm(range(0, num_samples, chunk_size)):
            end = min(i + chunk_size, num_samples)
            chunk = full_genotypes_array[i:end].astype(float)
            chunk[chunk == -1] = np.nan  # replace -1 with NaN for missing data

            # Scale the chunk by subtracting the *same* column means computed during fit
            scaled_chunk = np.where(np.isnan(chunk), 
                                    colmeans,  # Use precomputed colmeans instead of recalculating
                                    chunk)
            scaled_chunk -= colmeans

            # Transform the chunk using the fitted PCA
            pca_transformed_chunk = pca_obj.transform(scaled_chunk)

            # Save the PCA-transformed chunk to the HDF5 file
            transformed_dataset[i:end, :] = pca_transformed_chunk

    print(f"PCA transformation completed and saved to {pca_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PCA on genotype data")
    parser.add_argument('--plink_file_root', type=str, required=True, help='Path to Plink file (dont include extension)')
    parser.add_argument('--plink_file_name', type=str, required=True, help='Plink file name (e.g., extractedChrAllPruned.bed)')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder for PCA results')
    parser.add_argument('--subset', type=str, default='1KGP', help='Subset to filter and compute PCA on (default: 1KGP)')
    parser.add_argument('--pca_components', type=int, default=50, help='Number of PCA components')
    parser.add_argument('--chunk_size', type=int, default=5000, help='Chunk size for PCA transformation')

    args = parser.parse_args()

    main(args.plink_file_root, args.plink_file_name, args.output_folder, args.subset, args.pca_components, args.chunk_size)