import os
import argparse
import numpy as np
import pandas as pd
import data_loader
import manifold_methods
import plotting
import mappings

def main(args):

    # Load dataset and process according to type
    data, labels, color_dict = None, None, None
    if args.dataset_type == "1000G":
        inputs, class_labels, samples, snp_names, class_label_names = data_loader.load_data_1000G(args.data_path)
        labels = data_loader.preprocess_labels_1000G(class_labels, 
                                                     class_label_names)
        color_dicts = [mappings.pop_pallette_1000G_fine,
                       mappings.pop_pallette_1000G_coarse]
        label_orders = [mappings.label_order_1000G_fine, 
                        mappings.label_order_1000G_coarse]

    elif args.dataset_type == "covid":
        metadata = data_loader.load_metadata(args.metadata_path)
        embeddings = data_loader.load_metadata(args.data_path)
        
        merged = embeddings.merge(metadata, left_on='Unnamed: 0', right_on='sample_id')
        merged['collection_date'] = pd.to_datetime(merged['collection_date'], format='%Y-%m-%d')
        inputs = merged[merged.columns[['PC' in colname for colname in merged.columns]]].values

        labels = data_loader.preprocess_labels_sarscov2(merged)

        color_dicts = [mappings.pop_pallette_covid_fine,
                       mappings.pop_pallette_covid_coarse]
        label_orders = [mappings.label_order_covid_fine, 
                        mappings.label_order_covid_coarse]
    
    # Example usage:
    model_name, filename_prefix = plotting.generate_filenames(**vars(args))
    model_path = os.path.join('models', model_name)
    figure_path = os.path.join('figures', filename_prefix)

    # Perform manifold learning
    algo, transformed_data = None, None
    if args.manifold_algo == "pca":
        algo, transformed_data = manifold_methods.perform_pca(inputs, 
                                                              model_path,
                                                              n_components=args.components)
    elif args.manifold_algo == "tsne":
        algo, transformed_data = manifold_methods.perform_tsne(inputs, 
                                                               model_path,
                                                               n_components=2, 
                                                               perplexity=args.perplexity)
    elif args.manifold_algo == "phate":
        t = int(args.t) if args.t != 'auto' else 'auto'
        algo, transformed_data = manifold_methods.perform_phate(inputs, 
                                                                model_path,
                                                                knn=args.knn, 
                                                                decay=args.decay,
                                                                gamma=args.gamma,
                                                                t=t)

    # Plotting
    if args.plot:
        for label, color_dict, label_order, level in zip(labels,
                                                         color_dicts,
                                                         label_orders,
                                                         ["fine", "Coarse"]):

            plotting.plot_embeddings(transformed_data, 
                                     label, 
                                     figure_path + f"_{level}", 
                                     color_dict, 
                                     label_order,
                                     label_positions=args.label_positions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manifold learning on genetic data.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file")
    parser.add_argument("--metadata_path", type=str, default='', help="Path to the metadata data file (if exists)")
    parser.add_argument("--dataset_type", type=str, choices=["1000G", "covid"], 
                        required=True, help="Type of dataset: 'genetic' or 'metadata'")
    parser.add_argument("--manifold_algo", type=str, choices=["pca", "tsne", "phate"], 
                        required=True, help="Manifold learning algorithm to use")
    parser.add_argument("--components", type=int, default=2, help="Number of components for PCA/TSNE")
    parser.add_argument("--perplexity", type=int, default=30, help="Perplexity for TSNE")
    parser.add_argument("--early_exaggeration", type=float, default=12., help="Early exaggeration for TSNE")
    parser.add_argument("--knn", type=int, default=5, help="Number of nearest neighbors for PHATE")
    parser.add_argument("--decay", type=int, default=40, help="Decay parameter for PHATE")
    parser.add_argument("--gamma", type=int, default=1, help="gamma parameter for PHATE")
    parser.add_argument("--t", type=str, default='auto', help="t parameter for PHATE")
    parser.add_argument("--plot", action='store_true', help="Whether to plot embeddings")
    parser.add_argument("--label_positions", action='store_true', help="Whether to plot labels on the embeddings")
    args = parser.parse_args()

    main(args)
