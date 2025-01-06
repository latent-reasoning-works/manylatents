"""
Script to parse and save data into a hdf5 file format
"""
import argparse
import os
import multiprocessing as mp
import time
from datetime import datetime

import numpy as np
import h5py

# Load genotypes of a sample from a line in the genotype file
def parse_genotypes(line):
    # Line : Sample id and genotype values across all SNPs
    sample = (line.split('\t')[0]).strip()

    # Fill with genotypes of all SNPs for the individual
    genotypes = []
    for i in line.split('\t')[1:]:
        # Replace missing values with -1
        if i.strip() == './.' or i.strip() == 'NA':
            genotype = -1
        else:
            genotype = int(i.strip())

        genotypes.append(genotype)

    genotypes = np.array(genotypes, dtype='int8')

    return sample, genotypes


# Load labels of all samples from the label file
def load_labels(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    mat = np.array([l.strip('\n').split('\t') for l in lines])

    samples = mat[1:,0]
    labels = mat[1:,1]

    return samples, labels


# Order labels to match the provided list of samples
def order_labels(samples, samples_in_labels, labels):
    idx = [np.where(samples_in_labels == s)[0][0] for s in samples]

    return np.array([labels[i] for i in idx])


# Convert class labels to numerical values
def numeric_encode_labels(labels):
    label_names = np.sort(np.unique(labels))

    encoded_labels = [np.where(label_names==i)[0][0] for i in labels]

    return label_names, encoded_labels



def create_dataset():
    args = parse_args()
    start_time = time.time()

    #----------------------------------------------
    #               PARSE GENOTYPES
    #----------------------------------------------
    print('\n---')
    genotype_start_time = time.time()

    # Multiprocessing to load lines in parallel
    ncpus = args.ncpus
    print('Parsing genotypes using {} cpu(s)'.format(ncpus))
    pool = mp.Pool(processes=ncpus) #this starts ncpus nb of processes

    # Results returned by processes
    results = []

    # Every line is read by a different process
    with open(args.genotypes, 'r') as f:
        header_line = next(f)
        for line in f:
            # Add sample, sample's genotype accros all SNPs
            results.append(pool.apply_async(parse_genotypes, args=(line,)))

    # Get parsed samples and genotypes
    print('Parsed genotypes, getting the results')
    results_values = [p.get() for p in results]
    samples = np.array([i[0] for i in results_values])
    genotypes = np.array([i[1] for i in results_values])

    # Get snps names from header line
    snps = np.array([i.strip() for i in header_line.split('\t')[1:]], dtype='S')

    print('Parsed {} genotypes of {} samples in {} seconds'.format(
          len(snps), len(samples), time.time() - genotype_start_time))
    print('---\n')


    #----------------------------------------------
    #             LOAD CLASS LABELS
    #----------------------------------------------
    if args.class_labels is not None:
        # Loading class labels from file
        print('\n---\nLoading class labels')
        samples_in_labels, labels = load_labels(args.class_labels)
        print('Loaded {} class labels of {} samples'.format(
            len(labels), len(samples_in_labels)))

        # Matching class labels with genotypes
        print('\nMatching class labels and genotypes using sample ids')
        ordered_labels = order_labels(samples, samples_in_labels, labels)
        print('Matched genotypes and class labels of {} samples'.format(
            len(ordered_labels)))

        # Encode class labels as numerical values
        print('\nNumeric encoding of class labels')
        class_label_names, \
        encoded_class_labels = numeric_encode_labels(ordered_labels)
        print('Encoded {} classes: {}\n---\n'.format(
              len(class_label_names), class_label_names))


    #----------------------------------------------
    #           LOAD REGRESSION LABELS
    #----------------------------------------------
    if args.regression_labels is not None:
        # Loading regression labels from file
        print('\n---\nLoading regression labels')
        samples_in_labels, labels = load_labels(args.regression_labels)
        print('Loaded {} regression labels of {} samples'.format(
              len(labels), len(samples_in_labels)))

        # Matching regression labels with genotypes
        print('\nMatching regression labels and genotypes using sample ids')
        ordered_labels = order_labels(samples, samples_in_labels, labels)
        print('Matched genotypes and regression labels of {} samples'.format(
              len(ordered_labels)))

        # Convert regression labels to float
        regression_labels = ordered_labels.astype('float64')
        print('---\n')


    #----------------------------------------------
    #           WRITE DATASET TO FILE
    #----------------------------------------------
    # Dataset filename
    if args.out is not None:
        dataset_filename = args.out + '.hdf5'
    else:
        dataset_filename = 'dataset_' \
                + datetime.now().strftime("%Y_%m_%d_%Hh%Mmin%Ssec") + '.hdf5'

    # Dataset filename with full path
    dataset_fullpath = os.path.join(args.exp_path, dataset_filename)

    print('\n---\nSaving dataset to {}'.format(dataset_fullpath))

    # Create dataset
    f = h5py.File(dataset_fullpath, 'w')

    # Input features
    f.create_dataset('inputs', data=genotypes)
    # SNP names (Hdf5 doesn't support np UTF-8 encoding: snps.astype('S'))
    f.create_dataset('snp_names', data=snps.astype('S'))
    # Samples
    f.create_dataset('samples', data=samples.astype('S'))

    # Class labels
    if args.class_labels is not None:
        f.create_dataset('class_labels', data=encoded_class_labels)
        f.create_dataset('class_label_names', data=class_label_names.astype('S'))

    # Regression labels
    if args.regression_labels is not None:
        f.create_dataset('regression_labels', data=regression_labels)

    f.close()

    print('Program executed in {} seconds'.format(time.time()-start_time))
    print('---\n')


def parse_args():
    parser = argparse.ArgumentParser(
            description='Create hdf5 dataset from genotype and label files.'
            )

    # Directory where to save data
    parser.add_argument(
            '--exp-path',
            type=str,
            required=True,
            help='Path to directory where dataset will be written'
            )

    # Genotype file
    parser.add_argument(
            '--genotypes',
            type=str,
            required=True,
            help=('File of samples and their genotypes '
                  '(tab-separated format, one sample per line). '
                  'Missing genotypes can be encoded with NA, .\. or -1. '
                  'Provide full path')
            )

    # Classification labels (classification and regression tasks)
    parser.add_argument(
            '--class-labels',
            type=str,
            help=('File of samples and their class labels '
                  '(tab-separated format, one sample per line). '
                  'Provide full path')
            )

    # Regession labels (regression task)
    parser.add_argument(
            '--regression-labels',
            type=str,
            help=('File of samples and their regression labels '
                  '(tab-separated format, one sample per line). '
                  'Provide full path')
            )

    # Number of cpus for parallel loading of genotype file
    parser.add_argument(
            '--ncpus',
            type=int,
            default=1,
            help=('Number of cpus available to parse genotypes in parallel. '
                  'Default: %(default)i')
            )

    # Filename of returned dataset
    parser.add_argument(
            '--out',
            help=('Optional filename for the returned dataset. '
                  'If not provided the file will be named '
                  'dataset_date.hdf5')
            )

    return parser.parse_args()


if __name__ == '__main__':
    create_dataset()
