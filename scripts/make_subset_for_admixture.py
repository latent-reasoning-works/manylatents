import os
import numpy as np
import pandas as pd
import argparse

def main(metadata_path, admix_file_root):
    """
    Generate subset files for admixture analysis.

    Args:
        metadata_path (str): Path to metadata.
        admix_file_root (str): Path to the directory for admixture analysis.
    """

    # Load metadata
    metadata = pd.read_csv(metadata_path, sep=',', header=0, index_col=1)

    # Export IDs for admixture software
    filter_out_indices = metadata.query("~filter_pca_outlier & ~hard_filtered & ~filter_contaminated")

    # Global filtering
    global_samples_path = os.path.join(admix_file_root, 'tmp/global_samples_to_keep')
    filter_out_indices.index.to_series().to_csv(
        global_samples_path,
        index=True, header=False, sep=' '
    )

    # JUST AMR
    amr_samples_path = os.path.join(admix_file_root, 'tmp/AMR_ACB_ASW_samples_to_keep')
    indices_to_keep = filter_out_indices.query(
        "Genetic_region_merged == 'America' | Population == 'ASW' | Population == 'ACB'"
    )
    indices_to_keep.index.to_series().to_csv(
        amr_samples_path,
        index=True, header=False, sep=' '
    )

    # AMR+EUR+AFR
    amr_eur_afr_samples_path = os.path.join(admix_file_root, 'tmp/AMR_EUR_AFR_samples_to_keep')
    indices_to_keep = filter_out_indices.query(
        "Genetic_region_merged == 'America' | Genetic_region_merged == 'Europe' | Genetic_region_merged == 'Africa'"
    )
    indices_to_keep.index.to_series().to_csv(
        amr_eur_afr_samples_path,
        index=True, header=False, sep=' '
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare subsets for admixture analysis.")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to metadata.")
    parser.add_argument("--admix_file_root", type=str, required=True, help="Path to directory for admixture analysis.")

    args = parser.parse_args()
    main(args.metadata_path, args.admix_file_root)