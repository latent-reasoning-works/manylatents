import networkx as nx
import numpy as np

from utils import metrics


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
