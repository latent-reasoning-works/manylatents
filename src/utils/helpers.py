## FOR REFERENCE PURPOSES ONLY, BEING PHASED OUT

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#import tqdm
import phate
import scprep
from pyplink import PyPlink
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as svstack
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from tqdm import tqdm


def load_data(base_path, fname):
    # Load HGDP
    data_path = os.path.join(base_path, fname)

    pedfile = PyPlink(data_path)
    try:
        genotypes_array = np.load(os.path.join(base_path, '_raw_genotypes.npy'))
    except:
        genotypes_array = np.zeros([pedfile.get_nb_samples(), 
                                    pedfile.get_nb_markers()], 
                                   dtype=np.int8)

        for i, (marker_id, genotypes) in tqdm(enumerate(pedfile)):
            genotypes_array[:, i] = genotypes

        np.save(os.path.join(base_path, '_raw_genotypes.npy'), 
                genotypes_array)
    labels = pd.read_csv(os.path.join(base_path, 
                                      'gnomad.genomes.v3.1.2.hgdp_1kg_subset_sample_meta.reduced.tsv'), 
                         sep='\t')

    # remove duplicate info (appears in filter info too)
    labels = labels.drop(columns=['Project', 
                                  'Population', 
                                  'Genetic_region'])

    genotypes_array = genotypes_array[1:]  # remove first row
    labels = labels[1:]  # remove first row

    # Load filter data
    filter_info = pd.read_csv(os.path.join(base_path, 
                                           '4.3/gnomad_derived_metadata_with_filtered_sampleids.csv'), 
                              sep=',', 
                              index_col=1)

    merged_metadata = labels.set_index('sample').merge(filter_info, 
                                                       left_index=True, 
                                                       right_index=True)

    # load relatedness
    relatedness = pd.read_csv(os.path.join(base_path, 
                                           'HGDP+1KGP_MattEstimated_related_samples.tsv'), 
                              sep='\t', 
                              index_col=0)

    pop_palette_hgdp_coarse, pop_palette_hgdp_fine, _, _ = make_palette_label_order_HGDP(merged_metadata)
    
    return merged_metadata, relatedness, genotypes_array, (pop_palette_hgdp_coarse, 
                                                           pop_palette_hgdp_fine, 
                                                           None, 
                                                           None)


def make_palette_label_order_HGDP_old(populations, superpopulations):

    os.chdir('../../src')
    import mappings
    #import data_loader

    # SAS -> CSA + add MID, OCE
    pop_palette_hgdp_coarse = copy.deepcopy(mappings.pop_pallette_1000G_coarse)
    pop_palette_hgdp_coarse['CSA'] = mappings.pop_pallette_1000G_coarse['SAS']
    pop_palette_hgdp_coarse.pop('SAS')

    pop_palette_hgdp_coarse['MID'] = 'grey'
    pop_palette_hgdp_coarse['OCE'] = 'yellow'

    label_order_hgdp_coarse = copy.deepcopy(mappings.label_order_1000G_coarse)
    label_order_hgdp_coarse.remove('SAS')
    label_order_hgdp_coarse += ['CSA', 'MID', 'OCE']

    # Keep original 24/26 populations (with colors), and add new ones. New pops colored using superpop
    label_order_hgdp_fine = []
    for super_pop in np.unique(superpopulations):
        for pop in np.unique(populations[superpopulations==super_pop]):
            label_order_hgdp_fine.append(pop)

    # create tmp object to hold the original 26 populations
    mapping_26 = copy.deepcopy(mappings.pop_pallette_1000G_fine)
    mapping_26['GBR'] = mapping_26['CEUGBR']
    mapping_26['CEU'] = mapping_26['CEUGBR']
    mapping_26['STU'] = mapping_26['STUITU']
    mapping_26['ITU'] = mapping_26['STUITU']

    pop_palette_hgdp_fine = {}

    for super_pop in np.unique(superpopulations):
        for pop in np.unique(populations[superpopulations==super_pop]):
            if pop not in mapping_26.keys():
                # just use superpop color for now
                pop_palette_hgdp_fine[pop] = pop_palette_hgdp_coarse[super_pop]
            else:
                pop_palette_hgdp_fine[pop] = mapping_26[pop]

    return pop_palette_hgdp_coarse, pop_palette_hgdp_fine, label_order_hgdp_coarse, label_order_hgdp_fine

def make_palette_label_order_HGDP(metadata):
    # get color palette
    pop_pallette_1000G_coarse = {'East_Asia': 'blue',
                                'Europe': 'purple',
                                'America': 'red',
                                'Africa': 'green',
                                'Central_South_Asia': 'orange'
                               }
    label_order_1000G_fine = ['YRI', 'ESN', 'GWD', 'LWK', 'MSL', 'ACB', 'ASW',
                               'IBS',  'CEUGBR', 'TSI', 'FIN',
                               'PJL', 'BEB', 'GIH', 'STUITU',
                               'CHB', 'CHS', 'CDX', 'KHV', 'JPT',
                               'MXL', 'CLM', 'PEL', 'PUR']
    pop_colors=["#C7E9C0","#A1D99B","#74C476","#41AB5D","#238B45","#006D2C","#00441B",
                "#EFBBFF","#D896FF","#BE29EC","#800080",
                "#FEEDDE","#FDBE85","#FD8D3C","#E6550D",
                "#DEEBF7","#9ECAE1","#008080","#0ABAB5","#08519C",
               "#BC544B","#E3242B","#E0115F","#900D09","#7E2811"]
    pop_pallette_1000G_fine = {label:color for label,color in zip(label_order_1000G_fine, pop_colors)}

    pop_palette_hgdp_coarse = copy.deepcopy(pop_pallette_1000G_coarse)
    pop_palette_hgdp_coarse['Middle_East'] = 'grey'
    pop_palette_hgdp_coarse['Oceania'] = 'yellow'

    # create tmp object to hold the original 26 populations
    mapping_26 = copy.deepcopy(pop_pallette_1000G_fine)
    mapping_26['GBR'] = mapping_26['CEUGBR']
    mapping_26['CEU'] = mapping_26['CEUGBR']
    mapping_26['STU'] = mapping_26['STUITU']
    mapping_26['ITU'] = mapping_26['STUITU']

    pop_palette_hgdp_fine = {}
    superpopulations = metadata['Genetic_region_merged']
    populations = metadata['Population']

    for super_pop in np.unique(superpopulations):
        for pop in np.unique(populations[superpopulations==super_pop]):
            if pop not in mapping_26.keys():
                # just use superpop color for now
                pop_palette_hgdp_fine[pop] = pop_palette_hgdp_coarse[super_pop]
            else:
                pop_palette_hgdp_fine[pop] = mapping_26[pop]
    return pop_palette_hgdp_coarse, pop_palette_hgdp_fine, None, None

def replace_negative_one_with_nan(array):
    # Replace all occurrences of -1 with np.nan
    return np.where(array == -1, np.nan, array)

def compute_non_missing_overlap(non_missing_mask, recompute=False, save_path="non_missing_overlap.npz"):
    # Check if the result already exists
    if os.path.exists(save_path) and not recompute:
        print("Loading previously computed non-missing overlap matrix...")
        prev_comp = np.load(save_path)['overlap_matrix']
        return prev_comp

    # Convert non-missing mask to sparse format, treating False as 1 and True as 0
    sparse_mask = csr_matrix((~non_missing_mask).astype(int))

    # Initialize a list to store row-wise results
    results = []

    # Iterate over each row with tqdm for progress tracking
    for i in tqdm(range(sparse_mask.shape[0]), desc="Computing row-wise non-missing overlaps"):
        # Compute addition of row `i` with all rows in `sparse_mask`
        replicated_row = svstack([sparse_mask[i]] * sparse_mask.shape[0])

        # Count non-zero entries for each pair (row i + row j)
        nonzero_counts = (replicated_row+sparse_mask).getnnz(axis=1)

        # Append the non-zero counts as a sparse row to results
        results.append(nonzero_counts)

    # Stack all the result rows to form the final matrix
    final_result = np.vstack(results)
    final_result = len(non_missing_mask[0]) - final_result
    np.savez_compressed(save_path, overlap_matrix=final_result)

    return final_result

def hwe_normalize(genotypes_array):

    # Compute allele frequencies, ignoring NaNs
    allele_freqs = np.nanmean(genotypes_array / 2, axis=0)  # p = mean allele frequency

    # Center the matrix by subtracting 2 * allele frequency for each SNP
    centered_matrix = genotypes_array - 2 * allele_freqs

    # Compute Hardy-Weinberg variance for each SNP, avoiding division by zero
    hwe_variance = 2 * allele_freqs * (1 - allele_freqs)
    hwe_variance[hwe_variance == 0] = 1  # Avoid division by zero for monomorphic SNPs

    # Normalize each SNP by Hardy-Weinberg variance
    normalized_matrix = centered_matrix / np.sqrt(hwe_variance)
    return normalized_matrix

def preprocess_data_matrix(genotypes_array, recompute_overlap_counts=False):
    
    # Compute hwe normalized matrix
    genotypes_array = replace_negative_one_with_nan(genotypes_array)
    normalized_matrix = hwe_normalize(genotypes_array)

    # Create a mask for non-missing values
    non_missing_mask = ~np.isnan(genotypes_array)

    # Replace NaNs in the normalized matrix with zeros for compatibility with matrix multiplication
    normalized_matrix = np.where(non_missing_mask, normalized_matrix, 0)

    # speeds up computation by exploiting sparsity
    overlap_counts = compute_non_missing_overlap(non_missing_mask, recompute_overlap_counts)
    assert np.allclose(overlap_counts[:2], np.dot(non_missing_mask[0:2].astype(int), non_missing_mask.T))
    return normalized_matrix, overlap_counts

def approximate_kernel_random_projection(normalized_matrix, n_components=2000):
    projector = GaussianRandomProjection(n_components=n_components, random_state=42)
    reduced_genotype_matrix = projector.fit_transform(normalized_matrix)
    kernel_approx = reduced_genotype_matrix @ reduced_genotype_matrix.T
    return kernel_approx

def select_top_variance_snps(genotype_matrix, top_k=5000):
    # Compute variance across SNPs
    variances = np.var(genotype_matrix, axis=0)
    
    # Select top-k SNPs with the highest variance
    top_snp_indices = np.argsort(variances)[-top_k:]
    reduced_genotype_matrix = genotype_matrix[:, top_snp_indices]
    
    return reduced_genotype_matrix

# Compute a kernel on the reduced genotype matrix
def approximate_kernel_top_variance_snps(genotype_matrix, top_k=5000):
    reduced_genotype_matrix = select_top_variance_snps(genotype_matrix, top_k)
    kernel_approx = reduced_genotype_matrix @ reduced_genotype_matrix.T
    return kernel_approx

def compute_kernel_matrix(normalized_matrix, 
                          overlap_counts, 
                          approx='random_projection',
                          scale_by_overlap=True):

    if approx == 'random_projection':
        gram_approx = approximate_kernel_random_projection(normalized_matrix)
    elif approx == 'top_variance':
        gram_approx = approximate_kernel_top_variance_snps(normalized_matrix)
    elif approx == 'exact':
        gram_approx = normalized_matrix @ normalized_matrix.T

    #gt = normalized_matrix@normalized_matrix[0]
    #plt.scatter(gt, gram_approx1[:,0])
    #plt.xlim(-50000, 50000)
    #plt.ylim(-50000, 50000)
    
    if scale_by_overlap:
        gsm = gram_approx/overlap_counts
    else:
        gsm = gram_approx
    
    return gsm

def compute_pca_from_data_matrix(data,
                                 to_fit_on,
                                 to_transform_on,
                                 n_components=50):
    pca_emb = np.zeros((len(data), n_components))
    
    # SVD of data matrix (usual sklearn PCA implementation)
    pca_obj = PCA(n_components=n_components)
    pca_obj.fit(data[to_fit_on])

    # Transform on both
    pca_emb[to_fit_on] = pca_obj.transform(data[to_fit_on])
    pca_emb[to_transform_on] = pca_obj.transform(data[to_transform_on])
    return pca_emb, pca_obj

def compute_pca_from_kernel(gsm,
                            to_fit_on,
                            to_transform_on,
                            n_components=50):

    # Eigendecomposing kernel matrix

    # Step 1: Perform eigenvalue decomposition on the unrelated kernel subset
    eigenvalues, eigenvectors = np.linalg.eigh(gsm[to_fit_on, :][:, to_fit_on])

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices][:n_components]  # Select top NUM_PCS
    eigenvectors = eigenvectors[:, sorted_indices][:, :n_components]  # Select top NUM_PCS

    # Create PCA input matrix for all individuals
    pca_input = np.zeros((len(gsm), n_components))

    # Step 2: Project unrelated individuals into PCA space
    pca_input[to_fit_on] = eigenvectors * np.sqrt(eigenvalues)

    # Step 3: Project related individuals using the Nystrom extension
    kernel_related_unrelated = gsm[np.ix_(to_fit_on, to_transform_on)]
    pca_input[to_transform_on] = kernel_related_unrelated.T @ (eigenvectors / np.sqrt(eigenvalues + 1e-12))
    return pca_input, None


def compute_pca_from_hail(hail_pca_path, merged_metadata, num_pcs):
    pca_emb = pd.read_csv(hail_pca_path, sep='\t')
    to_return = merged_metadata.merge(pca_emb.set_index('s'), how='left', left_index=True, right_index=True)
    to_return = to_return[to_return.columns[to_return.columns.str.startswith('PC')].tolist()].values[:,:num_pcs]
    return to_return, None


def compute_phate(pca_input, to_fit_on, to_transform_on, **phate_params):
    phate_emb = np.zeros((len(pca_input), 2))

    # Step 4: Run PHATE on PCA-reduced data
    phate_operator = phate.PHATE(random_state=42, n_pca=None, **phate_params)
    
    # Fit PHATE on either the filtered unrelated or all data, based on phate_fit_related
    phate_operator.fit(pca_input[to_fit_on, :])
    
    # Transform all filtered individuals using PHATE embedding
    phate_emb[to_fit_on] = phate_operator.transform(pca_input[to_fit_on, :])
    phate_emb[to_transform_on] = phate_operator.transform(pca_input[to_transform_on, :])
    
    return phate_emb

def plot_pca_phate(pca_emb, phate_emb, indices_to_plot, pop_palette, pop_labels, ax=None):
    # Visualization
    if ax is not None:
        legend = False
    else:
        # no ax passed. Need to create figure
        fig, ax = plt.subplots(figsize=(20, 10), ncols=2, gridspec_kw={'wspace': 0.08})
        legend = True

    # PCA plot
    scprep.plot.scatter2d(
        pca_emb[indices_to_plot, :2],
        s=20,
        cmap=pop_palette,
        ax=ax[0],
        c=pop_labels[indices_to_plot],
        xticks=False,
        yticks=False,
        legend=True,
        legend_loc='lower center',
        legend_anchor=(0.5, -0.35),
        legend_ncol=8,
        label_prefix="PCA ",
        fontsize=8
    )
    ax[0].set_title("PCA of HGDP", fontsize=30)
    ax[0].get_legend().remove()

    # PHATE plot
    scprep.plot.scatter2d(
        phate_emb[indices_to_plot, :],
        s=20,
        cmap=pop_palette,
        ax=ax[1],
        c=pop_labels[indices_to_plot],
        xticks=False,
        yticks=False,
        legend=legend,
        legend_loc='lower center',
        legend_anchor=(0.5, -0.35),
        legend_ncol=8,
        label_prefix="PHATE ",
        fontsize=8
    )
    ax[1].set_title("PHATE of HGDP", fontsize=30)

def get_fit_transform_sets(fit_indices, transform_indices, fit_phate_on_both_sets):
    plot_set = fit_indices | transform_indices
    if fit_phate_on_both_sets:
        phate_fit_set = phate_trans_set = plot_set
    else:
        phate_fit_set = fit_indices
        phate_trans_set = transform_indices
    return phate_fit_set, phate_trans_set, plot_set

def plot_pca_phate_data_matrix(data,
                               fit_indices,
                               transform_indices,
                               pop_palette,
                               pop_labels,
                               fit_phate_on_both_sets=True,
                               ax=None,
                               num_pcs=50,
                               phate_params={'knn': 5, 't': 5}):
    pca_emb, pca_obj = compute_pca_from_data_matrix(data,
                                           fit_indices,
                                           transform_indices,
                                           n_components=num_pcs)
    
    phate_fit_set, phate_trans_set, plot_set = get_fit_transform_sets(fit_indices, 
                                                                      transform_indices, 
                                                                      fit_phate_on_both_sets)
        
    phate_emb = compute_phate(pca_emb, phate_fit_set, phate_trans_set, **phate_params)
    plot_pca_phate(pca_emb, phate_emb, plot_set, pop_palette, pop_labels, ax)
    return pca_emb, pca_obj

def plot_pca_phate_kernel_matrix(gsm,
                                 fit_indices,
                                 transform_indices,
                                 pop_palette,
                                 pop_labels,
                                 fit_phate_on_both_sets=True,
                                 ax=None,
                                 num_pcs=50,
                                 phate_params={'knn': 5, 't': 5}):
    pca_emb, _ = compute_pca_from_kernel(gsm,
                                      fit_indices,
                                      transform_indices,
                                      n_components=num_pcs)
    
    phate_fit_set, phate_trans_set, plot_set = get_fit_transform_sets(fit_indices, 
                                                                      transform_indices, 
                                                                      fit_phate_on_both_sets)

    phate_emb = compute_phate(pca_emb, phate_fit_set, phate_trans_set, **phate_params)
    plot_pca_phate(pca_emb, phate_emb, plot_set, pop_palette, pop_labels, ax)
    return pca_emb, None

def plot_pca_phate_hail(hail_pca_path,
                        merged_metadata,
                        fit_indices,
                        transform_indices,
                        pop_palette,
                        pop_labels,
                        fit_phate_on_both_sets=True,
                        num_pcs=50,
                        ax=None):
    
    pca_emb, _ = compute_pca_from_hail(hail_pca_path, merged_metadata, num_pcs)
    
    phate_fit_set, phate_trans_set, plot_set = get_fit_transform_sets(fit_indices, 
                                                                      transform_indices, 
                                                                      fit_phate_on_both_sets)

    phate_emb = compute_phate(pca_emb, phate_fit_set, phate_trans_set, knn=5, t=5)
    plot_pca_phate(pca_emb, phate_emb, plot_set, pop_palette, pop_labels, ax)
    return pca_emb, None   
