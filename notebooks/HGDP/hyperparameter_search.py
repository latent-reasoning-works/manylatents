import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import phate
from sklearn.manifold import TSNE
import tqdm
import argparse
import networkx as nx
import scprep

import metrics
import helpers

# Helper function to save/load objects
def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Compute PHATE embeddings or load if available
def compute_or_load_phate(pca_input, fit_idx, transform_idx, ts, phate_dir, **phate_params):
    os.makedirs(phate_dir, exist_ok=True)
    param_str = "_".join([f"{k}_{v}" for k, v in phate_params.items()])
    file_path = os.path.join(phate_dir, f"phate_{param_str}.pkl")

    if os.path.exists(file_path):
        print(f"Loading PHATE operator from {file_path}")
        phate_operator = load_pickle(file_path)
    else:
        phate_operator = phate.PHATE(random_state=42, **phate_params)
        phate_operator.fit(pca_input[fit_idx])
        save_pickle(phate_operator, file_path)
        print(f"Saved PHATE operator to {file_path}")

    output_embs = []
    for t in ts:
        phate_operator.set_params(t=t)
        phate_emb = np.zeros((len(pca_input), 2))
        phate_emb[fit_idx] = phate_operator.transform(pca_input[fit_idx])
        phate_emb[transform_idx] = phate_operator.transform(pca_input[transform_idx])
        output_embs.append(phate_emb)

    return output_embs, phate_operator

def compute_tsne(pca_input, fit_idx, transform_idx, **tsne_params):
    tsne_obj = TSNE(n_components=2, **tsne_params)
    tsne_emb = np.zeros(shape=(len(pca_input), 2))
    tsne_out = tsne_obj.fit_transform(pca_input[fit_idx | transform_idx])
    tsne_emb[fit_idx | transform_idx] = tsne_out
    return tsne_emb, tsne_obj

# Generate PHATE plots
def plot_phate_results(phate_embs_list, 
                       metadata, 
                       ts, 
                       param_values, 
                       param_name, 
                       cmap, 
                       output_dir):
    num_vals = len(phate_embs_list)
    num_t = len(phate_embs_list[0])

    # Ensure axes are always treated as 2D arrays
    fig, ax = plt.subplots(
        figsize=(10 * num_t, 10 * num_vals),
        nrows=num_vals,
        ncols=num_t,
        gridspec_kw={'wspace': 0.08},
        squeeze=False  # Ensure ax is always 2D
    )

    for i, embs in enumerate(phate_embs_list):
        for j, emb in enumerate(embs):
            scprep.plot.scatter2d(
                emb,
                s=5,
                ax=ax[i, j],
                c=metadata['Population'].values,
                cmap=cmap,
                xticks=False,
                yticks=False,
                legend=False,
                legend_loc='lower center',
                legend_anchor=(0.5, -0.35),
                legend_ncol=8,
                label_prefix="PHATE ",
                fontsize=8
            )
            ax[i, j].set_title('t={} {}={}'.format(ts[j], param_name, param_values[i]), fontsize=30)
    plt.tight_layout()
    plt.savefig(output_dir)
    plt.close()

# Compute quality metrics
def compute_quality_metrics(ancestry_coords, metadata, admixtures_k, admixture_ratios_list):
    to_keep = ~metadata['filter_pca_outlier'] & ~metadata['hard_filtered'] & ~metadata['filter_contaminated']
    ancestry_coords = ancestry_coords[to_keep]
    metadata = metadata[to_keep]
    admixture_ratios = [admixture_ratios_list_item[to_keep] for admixture_ratios_list_item in admixture_ratios_list]
    #admixture_ratios = admixture_ratios_list[3][to_keep]

    # geographic metrics
    metrics_dict = {
        "geographic_preservation": metrics.compute_geographic_metric(ancestry_coords, 
                                                                     metadata, 
                                                                     use_medians=False),
        "geographic_preservation_medians": metrics.compute_geographic_metric(ancestry_coords, 
                                                                             metadata, 
                                                                             use_medians=True),
        "geographic_preservation_far": metrics.compute_geographic_metric(ancestry_coords, 
                                                                         metadata, 
                                                                         use_medians=False, 
                                                                         only_far=True)
    }
    
    # admixture metrics
    for k, admixture_ratios_item in zip(admixtures_k, admixture_ratios):
        metrics_dict.update({
            "admixture_preservation_k={}".format(k): metrics.compute_continental_admixture_metric_dists(ancestry_coords, 
                                                                                                        admixture_ratios_item, 
                                                                                                        metadata, 
                                                                                                        use_medians=False),
            "admixture_preservation_medians_k={}".format(k): metrics.compute_continental_admixture_metric_dists(ancestry_coords, 
                                                                                                                admixture_ratios_item, 
                                                                                                                metadata, 
                                                                                                                use_medians=True),
            "admixture_preservation_far_k={}".format(k): metrics.compute_continental_admixture_metric_dists(ancestry_coords, 
                                                                                                            admixture_ratios_item, 
                                                                                                            metadata, 
                                                                                                            use_medians=False, 
                                                                                                            only_far=True),
            "admixture_preservation_laplacian_k={}".format(k): metrics.compute_continental_admixture_metric_laplacian(ancestry_coords, 
                                                                                                                      admixture_ratios_item),
        })

    return metrics_dict

def compute_pca_metrics(pca_input, emb, metadata):
    to_keep = ~metadata['filter_pca_outlier'] & ~metadata['hard_filtered'] & ~metadata['filter_contaminated']
    metrics_dict = {'pca_correlation': metrics.compute_pca_similarity(pca_input[to_keep], emb[to_keep])}

    return metrics_dict

def compute_topological_metrics(emb, metadata, phate_operator):
    # Adjacency matrix (iffusion operator, minus diagonal)
    A = phate_operator.diff_op - np.diag(phate_operator.diff_op)*np.eye(len(phate_operator.diff_op))
    graph = nx.from_numpy_array(A) # put into networkx
    component_Sizes = np.sort(np.array([len(k) for k in nx.connected_components(graph)]))[::-1]
    
    metrics_dict = {'connected_components': len(component_Sizes),
                    'component_sizes': component_Sizes}
    
    return metrics_dict


def load_data(admixtures_k, data_dir, admixture_dir):
    # Step -1: Load data
    merged_metadata, relatedness, genotypes_array, mapping_info = helpers.load_data()

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
        os.path.join(data_dir, 'pca_scores_hailcomputed.csv'),
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

# Helper to compute and append metrics
def compute_and_append_metrics(method_name, emb, pca_input, metadata, admixtures_k, admixture_ratios_list, hyperparam_dict, operator, results):
    # Compute metrics
    metrics_dict = compute_quality_metrics(emb, metadata, admixtures_k, admixture_ratios_list)
    
    # Add empty topological metrics if not computed
    if method_name in ["pca (2D)", "pca (50D)", "t-SNE"]:
        topological_dict = {'connected_components': None, 
                            'component_sizes': None}
    else:
        topological_dict = compute_topological_metrics(emb, metadata, operator)

    pca_metric_dict = compute_pca_metrics(pca_input, emb, metadata)

    metrics_dict.update(pca_metric_dict)
    metrics_dict.update(topological_dict)
    metrics_dict.update(hyperparam_dict)
    metrics_dict.update({'method': method_name})


    
    results.append(metrics_dict)

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

# Main function for hyperparameter search
def main(args):
    # Validate and create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, 'models')
    plot_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Load data
    admixtures_k = [2, 3, 4, 5, 6, 7, 8, 9]
    pca_input, metadata, fit_idx, transform_idx, admixture_ratios_list, cmap = load_data(
        admixtures_k, args.data_dir, args.admixture_dir
    )

    results = []

    # Compute for PCA, t-SNE

    # Compute quality metrics
    compute_and_append_metrics("pca (50D)", 
                               pca_input, 
                               pca_input, 
                               metadata, 
                               admixtures_k, 
                               admixture_ratios_list, 
                               {"gamma": "NA", "decay": "NA", "knn": "NA", "t": "NA"},
                               None,
                               results)
    
    compute_and_append_metrics("pca (2D)", 
                               pca_input[:, :2], 
                               pca_input, 
                               metadata, 
                               admixtures_k, 
                               admixture_ratios_list, 
                               {"gamma": "NA", "decay": "NA", "knn": "NA", "t": "NA"},
                               None,
                               results)

    tsne_emb, tsne_obj = compute_tsne(pca_input, fit_idx, transform_idx, init='pca')
    compute_and_append_metrics("t-SNE", 
                               tsne_emb, 
                               pca_input, 
                               metadata, 
                               admixtures_k, 
                               admixture_ratios_list, 
                               {"gamma": "NA", "decay": "NA", "knn": "NA", "t": "NA"},
                               None,
                               results)

    # Hyperparameters
    decays = [1, 10, 20, 40, 60, 120]
    knns = [5, 10, 20, 50, 100, 200]
    gammas = [0, 1]
    ts = [5, 10, 20, 50, 100]

    for gamma in gammas:
        for decay in decays:
            embeddings_list_k = []
            for knn in tqdm.tqdm(knns, desc=f"gamma={gamma}, decay={decay}"):
                # Compute PHATE embeddings
                embeddings_list, phate_operator = compute_or_load_phate(
                    pca_input,
                    fit_idx,
                    transform_idx,
                    ts,
                    model_dir,
                    n_landmark=None,
                    knn=knn,
                    decay=decay,
                    gamma=gamma
                )

                # Compute quality metrics
                for t, emb in tqdm.tqdm(zip(ts, embeddings_list), desc=f"quality metrics knn={knn}"):
                    compute_and_append_metrics("phate", 
                                               emb, 
                                               pca_input, 
                                               metadata, 
                                               admixtures_k, 
                                               admixture_ratios_list, 
                                               {"gamma": gamma, "decay": decay, "knn": knn, "t": t},
                                               phate_operator,
                                               results)

                embeddings_list_k.append(embeddings_list)

            # Save plots    
            plot_phate_results(
                embeddings_list_k,
                metadata,
                ts,
                knns,
                "knn",
                cmap,
                os.path.join(plot_dir, f"results_gamma={gamma}_decay={decay}.png")
            )

    # Save metrics to CSV
    results_df = pd.DataFrame(results)
    fname = os.path.join(args.output_dir, "results.csv")
    results_df.to_csv(fname, index=False)
    print(f"Results saved to {fname}")


# Argparse for cleaner CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PHATE Hyperparameter Search")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save PHATE results and plots")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing PCA data")
    parser.add_argument("--admixture_dir", type=str, required=True, help="Directory containing admixture data")
    args = parser.parse_args()

    main(args)