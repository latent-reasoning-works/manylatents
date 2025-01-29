import os
import sys
import numpy as np
import pandas as pd
#import hail as hl
import argparse

def fix_array(arr):
    """
    Convert True/False to boolean while preserving NaNs.
    """
    return np.where(arr == True, True, np.where(arr == False, False, np.nan))

def main(data_root):
    """
    Process gnomAD metadata and create derived metadata with filtering information.

    Args:
        data_root (str): Path to the data directory.
    """
    # Load gnomAD metadata
    gnomad_meta_path = os.path.join(data_root, 'gnomad_meta_updated.tsv')
    gnomad_meta = pd.read_csv(gnomad_meta_path, sep='\t')

    # Identify potential filter columns
    filter_columns = gnomad_meta.columns[gnomad_meta.columns.str.contains('filter|exclude')]
    true_filters = []

    # Process each filter column
    for col in filter_columns:
        try:
            unique_vals = np.unique(gnomad_meta[col])
            if len(unique_vals) > 3:
                gnomad_meta[col] = fix_array(gnomad_meta[col])
            _, counts = np.unique(gnomad_meta[col], return_counts=True)
            if np.max(counts) < 4150:
                true_filters.append(col)
        except Exception as e:
            print(f"Error processing column {col}: {e}")

    # Hard-filtered samples
    hard_filtered_samples = gnomad_meta.loc[
        gnomad_meta['sample_filters.hard_filtered'].apply(lambda x: x == True), 'project_meta.sample_id'
    ]
    assert len(hard_filtered_samples) == 31

    # Contaminated samples
    contaminated_samples = ['HGDP01371', 'LP6005441-DNA_A09']
    filtered_samples = hard_filtered_samples.tolist() + contaminated_samples

    # Load related samples from Hail Table
    # hl.init(spark_conf={'spark.driver.memory': '4g', 'spark.executor.memory': '4g'})
    # related_samples_path = os.path.join(data_root, 'related_sample_ids.ht')
    # related_samples = hl.read_table(related_samples_path).to_pandas()['node'].tolist()
    # hl.stop()
    related_samples = pd.read_csv(os.path.join(data_root,
                                               'HGDP+1KGP_related_samples.csv'))['node'].tolist()

    # Load PCA outliers
    pca_outliers_path = os.path.join(data_root, 'pca_outliers.txt')
    pca_outliers = pd.read_csv(pca_outliers_path, header=None, names=['sample_id'])['sample_id'].tolist()

    # Create derived metadata
    gnomad_derived_metadata = gnomad_meta[['project_meta.sample_id']].copy()

    # Add filtering information
    gnomad_derived_metadata['filter_pca_outlier'] = gnomad_derived_metadata['project_meta.sample_id'].isin(pca_outliers)
    gnomad_derived_metadata['filter_king_related'] = gnomad_derived_metadata['project_meta.sample_id'].isin(related_samples)
    gnomad_derived_metadata['hard_filtered'] = gnomad_derived_metadata['project_meta.sample_id'].isin(hard_filtered_samples)
    gnomad_derived_metadata['filter_contaminated'] = gnomad_derived_metadata['project_meta.sample_id'].isin(contaminated_samples)

    # Add population and region information
    gnomad_derived_metadata['Population'] = gnomad_meta['hgdp_tgp_meta.Population']
    gnomad_derived_metadata['Genetic_region'] = gnomad_meta['hgdp_tgp_meta.Study.region']

    # Merge regions
    genetic_region_mapping = {
        'AFR': 'Africa',
        'EAS': 'East_Asia',
        'SAS': 'Central_South_Asia',
        'EUR': 'Europe',
        'AMR': 'America'
    }
    gnomad_derived_metadata['Genetic_region_merged'] = gnomad_derived_metadata['Genetic_region'].replace(genetic_region_mapping)

    # Add project information
    gnomad_derived_metadata['Project'] = 'NA'
    gnomad_derived_metadata.loc[gnomad_meta['subsets.hgdp'], 'Project'] = 'HGDP'
    gnomad_derived_metadata.loc[gnomad_meta['subsets.tgp'], 'Project'] = '1KGP'

    # Add latitude and longitude
    gnomad_derived_metadata[['latitude', 'longitude']] = gnomad_meta[['latitude', 'longitude']]

    # Export derived metadata
    output_path = os.path.join(data_root, 'gnomad_derived_metadata_with_filtered_sampleids.csv')
    gnomad_derived_metadata.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process gnomAD metadata and create derived metadata with filtering information.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the data directory.")

    args = parser.parse_args()
    main(args.data_root)