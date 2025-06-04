import os
import pandas as pd
from pyplink import PyPlink
import argparse

def main(metadata_path, admix_file_root):

    # Read the CSV file (to get ID order)
    ukbb_csv = pd.read_csv(metadata_path)

    # load fam files to get ID order for train/test
    fam_train = pd.read_csv(os.path.join(admix_file_root, "random10K_EUR_and_others.fam"),
                            header=None, 
                            sep=' ').rename(columns={0: 'ID'})
    fam_test = pd.read_csv(os.path.join(admix_file_root, "random10K_EUR_and_others_unseen.fam"),
                           header=None, 
                           sep=' ').rename(columns={0: 'ID'})

    # Open metadata file
    metadata = pd.read_csv(metadata_path)
    for k in range(2,10):

        # Load Q files
        Qfile_train =  pd.read_csv(os.path.join(admix_file_root, 'neuralAdmixture.{}.Q'.format(k)),
                                   sep=' ', 
                                   header=None)
        Qfile_test =  pd.read_csv(os.path.join(admix_file_root, 'random10K_EUR_and_others_unseen.{}.Q'.format(k)),
                                  sep=' ', 
                                  header=None)

        Qfile_train = pd.concat([Qfile_train, fam_train['ID']], axis=1)
        Qfile_test = pd.concat([Qfile_test, fam_test['ID']], axis=1)
        Qfile = pd.concat([Qfile_train, Qfile_test])
        Qfile = pd.merge(Qfile, ukbb_csv[['IDs', 'Population', 'self_described_ancestry']], 
                         left_on='ID', 
                         right_on='IDs')
        
        # put in correct order
        final_df = Qfile.set_index('ID').loc[ukbb_csv['IDs']].drop(columns='IDs')
        #final_df = pd.concat([metadata['IDs'], Qfile, metadata[['self_described_ancestry', 'pop']]], axis=1)

        # Save final dataframe
        output_file = os.path.join(admix_file_root, 'UKBB.{}_metadata.tsv'.format(k))
        final_df.to_csv(output_file, index=False, header=False, sep='\t')
        print('finished: {}'.format(output_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up and process admixture output files.")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to metadata.")
    parser.add_argument("--admix_file_root", type=str, required=True, help="Path to directory for admixture analysis.")

    args = parser.parse_args()
    main(args.metadata_path, args.admix_file_root)