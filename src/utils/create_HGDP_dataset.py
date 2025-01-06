import os
import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.metrics import pairwise_distances
from sklearn import decomposition

import mappings
import data_loader


def load_admix_ratio(sample_ids_path, admix_ratios_path, samples, labels):
    admix_labels = np.zeros([len(samples), 3])
    admix_labels = pd.DataFrame(admix_labels, index=samples)

    sample_ids = pd.read_csv(sample_ids_path, sep=' ', header=None)
    admix_ratios = pd.read_csv(admix_ratios_path, sep=' ', header=None)

    admix_ratios = admix_ratios.set_index(sample_ids[0])
    admix_labels.update(admix_ratios)

    # Old less efficient approach
    #admix_df = pd.concat([admix_ratios, sample_ids], axis=1)
    #admix_df.columns = ['admix 1', 'admix 2', 'admix 3' , 'sample id']
    

    #matching_indices = np.array([np.where(admix_df['sample id'][j] == samples)[0][0] if len(np.where(admix_df['sample id'][j] == samples)[0]) > 0 else -1 for j in range(len(admix_df))])
    #for i, idx in enumerate(matching_indices):
    #    if idx != -1:
    #        admix_labels[idx] = admix_df.iloc[i].values[:3]

    # color in all EUR or AFR individuals as 1
    #admix_labels_inc_EURAFR = admix_labels
    #admix_labels_inc_EURAFR[labels == 'EUR', 2] = 1
    #admix_labels_inc_EURAFR[labels == 'AFR', 0] = 1

    return admix_labels, None


def main(args):
    print('Creating 1000G dataset...')
    inputs, class_labels, sample, snp_names, class_label_names, metadata_labels =\
    data_loader.load_data_HDGP(args.hdf5_data_file, 
                               args.metadata_path, 
                               args.unrelated_sampleid_1000G_path)
    
    pca_obj = sklearn.decomposition.PCA(n_components=100, random_state=42)
    pca_input = pca_obj.fit_transform(inputs)

    pd.DataFrame(pca_input, 
                 index=sample).to_csv(os.path.join(args.output_file_path, 
                                                    'HGDP_PCA'))

    pd.DataFrame({'population': metadata_labels['Population'],
                  'superpopulation': metadata_labels['Superpopulation']}, 
                 index=sample).to_csv(os.path.join(args.output_file_path, 
                                                    'HGDP_labels'))

    admix_labels, admix_labels_inc_EURAFR = load_admix_ratio(args.admix_sampleid_file, 
                                                             args.admix_ratios_file,
                                                             sample,
                                                             class_labels)

    pd.DataFrame(admix_labels, 
                 index=sample).rename(columns={0: "African ancestry (%)", 
                                                1: "Amerindigenous ancestry (%)", 
                                                2: "European ancestry (%)"}
                                                    ).to_csv(os.path.join(args.output_file_path, 
                                                                          'HGDP_admix_ratios'))

    pd.DataFrame(admix_labels_inc_EURAFR, 
                 index=sample).rename(columns={0: "African ancestry (%)", 
                                                1: "Amerindigenous ancestry (%)",\
                                                2: "European ancestry (%)"}
                                      ).to_csv(os.path.join(args.output_file_path,
                                                            'HGDP_admix_ratios_inc_EURAFR'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loads HGDP data from HD5F dataset")
    parser.add_argument("--hdf5_data_file", 
                        type=str, 
                        required=True, 
                        help="Path to the hdf5 data file")
    parser.add_argument("--metadata_path", 
                        type=str, 
                        required=True, 
                        help="Path to the metadata file")
    parser.add_argument("--unrelated_sampleid_1000G_path", 
                        type=str, 
                        required=True, 
                        help="Path to the unrelated samples file")
    parser.add_argument("--admix_sampleid_file", 
                        type=str, 
                        default='', 
                        help="Path to the admixture sample ids file")
    parser.add_argument("--admix_ratios_file", 
                        type=str, 
                        default='', 
                        help="Path to the admixture ratios file")
    parser.add_argument("--output_file_path", 
                        type=str, 
                        default='', 
                        help="Where to save output files")

    args = parser.parse_args()

    main(args)
