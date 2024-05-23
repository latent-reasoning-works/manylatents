import os
import argparse
import numpy as np
import pandas as pd
import data_loader
import manifold_methods
import plotting
import mappings
import os

import h5py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns
import scipy
import sklearn
from sklearn import datasets
from sklearn.metrics.pairwise import pairwise_kernels
import graphtools
#from pydiffmap.diffusion_map import DiffusionMap
import phate
import tqdm

from graph_utils import get_graph_representation, plot_graph, get_groups_from_graph

def main(args):
    exp_path = '/lustre06/project/6065672/shared/DietNet/1KGB_POP24/1KGP/WGS30X_V1/oldnow/MattsPlace'
    fname = '1000G.2504_WGS30x.GSA17k_MHI.intersectGSA.miss10perc.maf0.05.pruned.autosomes.noHLA.phased_imputed_V1.hdf5'

    with h5py.File(os.path.join(exp_path, fname), 'r') as hf:
        #model_attrs = hf['gradients'][:]
        #print('loaded gradient of fc1 w.r.t. input from {}'.format(attr_fc1_saliency_name))
        inputs = hf['inputs'][:]
        class_label_names = hf['class_label_names'][:]
        class_labels = hf['class_labels'][:]
        samples = hf['samples'][:]
        snp_names = hf['snp_names'][:]

    # make labels
    label_with_names = [str(class_label_names[y])[2:-1] for y in class_labels]
    class_label_names = [str(label)[2:-1] for label in class_label_names]
    label_with_superpop_names = np.zeros_like(label_with_names)
    for label in mappings.super_pops_1000G:
        index = pd.DataFrame(label_with_names).isin(mappings.super_pops_1000G[label]).values.flatten()
        label_with_superpop_names[index] = label
    
    # Compute PCA first (most expensive step)
    pca_obj = sklearn.decomposition.PCA(n_components=100, random_state=42)
    pca_input = pca_obj.fit_transform(inputs)
    
    # compare to distances in underlying space
    ambient_distances = sklearn.metrics.pairwise_distances(pca_input)
    
    # PCA 1,2 distances
    pca_distances = sklearn.metrics.pairwise_distances(pca_input[:,:2])
    
    # subset of distances
    r_idx = np.random.choice(len(pca_distances.flatten()), 5000, replace=False)

    #phate_embeddings = []
    #phate_ops = []
    knns = []
    alphas = []
    pca_dist_corrs = []
    amb_dist_corrs = []
    num_edges_graphs = []
    num_groups = []
    group_sizes = []
    ts = []
    t_auto_vals = []
    
    for knn in tqdm.tqdm([5, 10, 25, 100, 150]):
        for alpha in [0.01, 0.25, 0.5, 1, 10, 20, 40, 60, 100]:
            for t in [1, 5, 'auto']:
                knns.append(knn)
                alphas.append(alpha)
                phate_operator = phate.PHATE(random_state=42, n_pca=None, knn=knn, decay=alpha, t=t)
                phate_operator.fit(pca_input)
                phate_emb = phate_operator.transform()
                #phate_ops.append(phate_operator)
                #phate_embeddings.append(phate_emb)

                # Collect data
                phate_distances = sklearn.metrics.pairwise_distances(phate_emb)
                pca_dist_corr, _ = scipy.stats.spearmanr(phate_distances.flatten()[r_idx], pca_distances.flatten()[r_idx])
                amb_dist_corr, _ = scipy.stats.spearmanr(phate_distances.flatten()[r_idx], ambient_distances.flatten()[r_idx])
                amb_dist_corrs.append(amb_dist_corr)
                pca_dist_corrs.append(pca_dist_corr)

                # Topological info
                K = np.array(phate_operator.graph.K.todense())
                num_edges_graph = (K>0).sum()
                indices = get_groups_from_graph(K)
                group, sizes = np.unique(indices, return_counts=True)
                num_connected = len(group)

                num_edges_graphs.append(num_edges_graph)
                num_groups.append(num_connected)
                group_sizes.append(sizes)
                ts.append(t)
                
                if t == 'auto':
                    t_auto_vals.append(phate_operator.optimal_t)
                else:
                    t_auto_vals.append(np.nan)

            dataset = pd.DataFrame({'knn': knns, 
                                    'alpha': alphas, 
                                    'pca_dist_corrs': pca_dist_corrs,
                                    'amb_dist_corrs': amb_dist_corrs,
                                    'num_edges_graphs': num_edges_graphs,
                                    'num_groups': num_groups,
                                    'group_sizes': group_sizes,
                                    't': ts,
                                    't_auto_val': t_auto_vals})
            dataset.to_csv('phate_1000G_data.csv')
            print('Dataset saved to: {}'.format('phate_1000G_data.csv'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manifold learning check PCA Similarity")
    args = parser.parse_args()

    main(args)
