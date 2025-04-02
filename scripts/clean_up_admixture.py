import os
import pandas as pd
from pyplink import PyPlink
import argparse

def main(plink_prefix, metadata_path, admix_file_root):
    """
    Clean up and process admixture output files by adding metadata.

    Args:
        metadata_path (str): Path to metadata.
        admix_file_root (str): Path to directory for admixture analysis.
    """

    # Load metadata
    metadata = pd.read_csv(metadata_path, sep=',', header=0, index_col=1)

    # Define prefixes for subsets
    prefixes = [
        'AMR_ACB_ASW',
        'AMR_EUR_AFR',
        'AMR_ACB_ASW_1KGP_ONLY',
        'global'
    ]

    # Fix admixture files
    for k in range(2, 10):  # Components from 2 to 9
        for prefix in prefixes:
            fname = os.path.join(admix_file_root, f'{prefix}.{k}.Q')
            if not os.path.exists(fname):
                print(f'Could not load {fname}. Skipping...')
                continue

            # Load admixture ratios
            admixture_ratios = pd.read_csv(fname, header=None, sep=' ')

            # Get label order directly from plink file
            pedfile = PyPlink(plink_prefix)
            sample_id = pedfile.get_fam()[['fid', 'iid']].rename(columns={'iid': 'sample_id', 'fid': 'sample_fid'})

            # Merge with population metadata
            pop_df = metadata[['Population', 'Genetic_region_merged']].reset_index()
            final_df = pd.concat([sample_id, admixture_ratios], axis=1)
            final_df = pd.merge(
                left=final_df, 
                right=pop_df, 
                left_on='sample_id', 
                right_on='project_meta.sample_id',
                how='left'
            )
            final_df = final_df.drop(columns=['sample_fid', 'project_meta.sample_id'])

            # Save final dataframe
            output_file = os.path.join(admix_file_root, f'{prefix}.{k}_metadata.tsv')
            final_df.to_csv(output_file, index=False, header=False, sep='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up and process admixture output files.")
    parser.add_argument("--plink_prefix", type=str, required=True, help="Path and prefix of plink file")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to metadata.")
    parser.add_argument("--admix_file_root", type=str, required=True, help="Path to directory for admixture analysis.")

    args = parser.parse_args()
    main(args.plink_prefix, args.metadata_path, args.admix_file_root)