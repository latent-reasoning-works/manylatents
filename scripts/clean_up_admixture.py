import os
import pandas as pd
from pyplink import PyPlink
import argparse

def main(indices_file, metadata_path, admix_file_root):
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

            # Get label order from indices_file
            sample_id = pd.read_csv(indices_file, header=None, sep=' ')

            sample_id = sample_id.rename(columns={0: 'sample_id'})
            sample_id = sample_id.drop(columns=[1])

            # Merge with population metadata
            pop_df = metadata[['Population', 'Genetic_region_merged']].reset_index()
            final_df = pd.concat([sample_id, admixture_ratios], axis=1)
            
            final_df = pd.merge(
                left=pop_df, 
                right=final_df, 
                left_on='project_meta.sample_id', 
                right_on='sample_id',
                how='left'
            )

            final_df = final_df.drop(columns=['sample_id'])
            final_df = final_df.rename(columns={'project_meta.sample_id': 'sample_id'})
            all_but_last_2_cols = [col for col in final_df.columns if col not in ['Population', 
                                                                                  'Genetic_region_merged']]
            final_df = final_df[all_but_last_2_cols + ['Population', 'Genetic_region_merged']]

            # Save final dataframe
            output_file = os.path.join(admix_file_root, f'{prefix}.{k}_metadata.tsv')
            final_df.to_csv(output_file, index=False, header=False, sep='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up and process admixture output files.")
    parser.add_argument("--indices_file", type=str, required=True, help="File containing indices passed to admixture software")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to metadata.")
    parser.add_argument("--admix_file_root", type=str, required=True, help="Path to directory for admixture analysis.")

    args = parser.parse_args()
    main(args.indices_file, args.metadata_path, args.admix_file_root)