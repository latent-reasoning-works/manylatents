import os
import pandas as pd
from pyplink import PyPlink
import argparse

def main(metadata_path, admix_file_root):
    # Open metadata file
    metadata = pd.read_csv(metadata_path)
    for k in range(2,10):
        Qfile =  pd.read_csv(os.path.join(admix_file_root, 'ukb_k{}Qhat.txt.transpose'.format(k)),
                             sep=' ', header=None)
        final_df = pd.concat([metadata['IDs'], Qfile, metadata[['self_described_ancestry', 'pop']]], axis=1)

        # Save final dataframe
        output_file = os.path.join(admix_file_root, 'UKBB.{}_metadata.tsv'.format(k))
        final_df.to_csv(output_file, index=False, header=False, sep='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up and process admixture output files.")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to metadata.")
    parser.add_argument("--admix_file_root", type=str, required=True, help="Path to directory for admixture analysis.")

    args = parser.parse_args()
    main(args.metadata_path, args.admix_file_root)