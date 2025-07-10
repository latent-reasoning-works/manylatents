#!/usr/bin/env python3
"""
prep_aou_files.py
----------------------

1. Fit PCA on --fit-prefix (BED/BIM/FAM prefix).
2. Project BOTH cohorts:
      a) reference  → --fit-out
      b) target     → --transform-out
3. Save each as CSV with columns dim_1 … dim_N (no zip).

Example
-------
python fit_and_project_pca.py \
  --fit-prefix       /data/hgdp/hgdp            \
  --metadata-csv     /data/hgdp/meta.tsv        \
  --fit-out          hgdp_pca_n_components=50.csv \
  --transform-out    ukbb_pca_n_components=50.csv \
  --n-components     50 --chunk-size 5000
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import tqdm
import numpy as np
import pandas as pd
from pyplink import PyPlink
from sklearn.decomposition import PCA



def plink_to_numpy(prefix: Path) -> np.ndarray:
    """Return (samples × SNPs) int8 genotype matrix, caching to *_raw_genotypes.npy"""
    cache = prefix.parent / (prefix.name + "_raw_genotypes.npy")
    if cache.exists():
        return np.load(cache, mmap_mode="r")
    ped = PyPlink(str(prefix))
    G = np.zeros((ped.get_nb_samples(), ped.get_nb_markers()), dtype=np.int8)
    for j, (_, g) in tqdm.tqdm(enumerate(ped), total=ped.get_nb_markers(), desc="reading PLINK", file=sys.stdout):
        G[:, j] = g
    np.save(cache, G)
    return G


def replace_minus_one(arr):  # -1 → NaN
    return np.where(arr == -1, np.nan, arr)


def hw_norm(geno):
    p = np.nanmean(geno / 2, axis=0)
    var = 2 * p * (1 - p)
    var[var == 0] = 1.0
    centred = geno - 2 * p
    return centred / np.sqrt(var), 2 * p, var


def preprocess(geno):
    geno = replace_minus_one(geno)
    norm, mu, var = hw_norm(geno)
    norm = np.where(np.isnan(geno), 0, norm)
    return norm, mu, var


def save_csv(path: Path, mat: np.ndarray):
    cols = [f"dim_{i}" for i in range(1, mat.shape[1] + 1)]
    pd.DataFrame(mat, columns=cols).to_csv(path, index=False)
    print(f"✔  wrote {path}   ({mat.shape[0]} rows)")

## Metadata files
def load_tsv_file(name_of_file_in_bucket, path_to_file_in_bucket, path_to_local_file):

    if os.path.exists(os.path.join(path_to_local_file, name_of_file_in_bucket)):
        print(f'[INFO] {name_of_file_in_bucket} already downloaded into your working space')
    else:
        # get the bucket name
        my_bucket = os.getenv('WORKSPACE_BUCKET')
        # copy csv file from the bucket to path_to_local_file
        os.system(f"gsutil cp '{my_bucket}/{path_to_file_in_bucket}/{name_of_file_in_bucket}' '{path_to_local_file}'")
        print(f'[INFO] {name_of_file_in_bucket} is successfully downloaded into your working space')

    # save dataframe in a csv file in the same workspace as the notebook
    local_path = os.path.join(path_to_local_file, name_of_file_in_bucket)
    my_dataframe = pd.read_csv(local_path, sep='\t')
    return my_dataframe

## Load genotype data
def load_data(fname, path_to_file_in_bucket, path_to_local_file):
    tar_fname = os.path.join(path_to_local_file, fname + '.tar.gz')
    if os.path.exists(tar_fname):
        print(f'[INFO] {tar_fname} already downloaded into your working space')
    else:
        # get the bucket name
        my_bucket = os.getenv('WORKSPACE_BUCKET')

        # copy csv file from the bucket to path_to_local_file
        try:
            subprocess.run(
                ["gsutil", "cp", f"{my_bucket}/{path_to_file_in_bucket}/{fname}.tar.gz", path_to_local_file],
                check=True
            )
            print(f"[INFO] {fname}.tar.gz successfully downloaded")
        except subprocess.CalledProcessError:
            print(f"[ERROR] Failed to download {fname}.tar.gz from bucket")

    bed_path = Path(path_to_local_file) / (fname + ".bed")
    if not os.path.exists(bed_path):
        if not os.path.exists(tar_fname):
            print(f"{tar_fname} not found after attempted download")
        subprocess.run(["tar", "-xvf", tar_fname, "-C", path_to_local_file], check=True)

def replace_for_reference(lst):
    return [item.replace("forReference", "") if item.startswith("forReference") else item for item in lst]

def make_metadata_file(plink_prefix, path_to_metadata_files):
    # get metadata file
    demographic_data = load_tsv_file('DemographicData.tsv',
                                     'data/V2/Metadata',
                                     path_to_metadata_files)
    sire_data = load_tsv_file('ProcessedSIRE_V7.tsv',
                              'data/V2/Metadata',
                              path_to_metadata_files)

    pedfile = PyPlink(os.path.join(path_to_metadata_files,
                                   plink_prefix))

    all_samples = pedfile.get_fam()
    all_markers = pedfile.get_bim()
    
    all_samples["iid"] = all_samples["iid"].astype(str).str.strip()
    sire_data["person_id"] = sire_data["person_id"].astype(str).str.strip()

    metadata = pd.merge(
        all_samples, sire_data,
        left_on="iid", right_on="person_id", how="left"
    )
    
    metadata = metadata.drop(columns=['father', 'mother', 'gender', 'status'])
    
    demographic_data["person_id"] = demographic_data["person_id"].astype(str).str.strip()

    metadata = pd.merge(
        metadata, demographic_data,
        left_on="person_id", right_on="person_id", how="left"
    )
    
    # conform to ManyLatent expected columns
    reference_subset = all_samples.fid != 'AOU'

    metadata['Population'] = metadata['fid']
    metadata.loc[reference_subset, 'Population'] = replace_for_reference(all_samples.loc[reference_subset, 'fid'])

    metadata = metadata.rename(columns={'iid': 'sample_id'})

    metadata['related'] = True

    metadata.to_csv(os.path.join(path_to_metadata_files, 'aou_metadata.csv'))
    
    return metadata

# --------------------------------------------------------------------------- #
def main(a):

    # get plink files
    load_data(a.plink_prefix, 
              'Data/V2/1KGPHGDPAOU_V7', 
              a.path_to_metadata_files)

    # get metadata
    metadata = make_metadata_file(a.plink_prefix,
                                  a.path_to_metadata_files)

    # ------------------ load genotypes for fit-cohort ----------------------- #
    plink_path = Path(a.path_to_metadata_files) / a.plink_prefix
    G = plink_to_numpy(plink_path)

    print(f'loaded {os.path.join(a.path_to_metadata_files, a.plink_prefix)}')
    
    reference_subset = metadata.fid != 'AOU'
    G_fit = G[reference_subset]
    G_tr = G[~reference_subset]

    G_fit, mu, var = preprocess(G_fit)

    print(f" Fitting PCA on {reference_subset.sum()} / {len(reference_subset)} samples")
    pca = PCA(n_components=a.n_components)
    pca.fit(G_fit)

    G_fit_pca = pca.transform(G_fit)
    save_csv(Path(a.fit_out), G_fit_pca)

    # ------------------ project every row of second cohort ------------------ #
    n, k = G_tr.shape[0], a.chunk_size
    G_tr_pca = np.empty((n, a.n_components), dtype=np.float32)

    for i in tqdm.tqdm(range(0, n, k), desc="projecting second cohort", file=sys.stdout):
        sl = slice(i, min(i + k, n))
        chunk = G_tr[sl].astype(float)
        chunk[chunk == -1] = np.nan
        chunk = np.where(np.isnan(chunk), 0, (chunk - mu) / np.sqrt(var))
        G_tr_pca[sl] = pca.transform(chunk)

    save_csv(Path(a.transform_out), G_tr_pca)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="QC-aware PCA fit + project 2nd PLINK cohort")
    p.add_argument("--plink-prefix",     required=True, help="PLINK prefix used for fitting and transforming PCA")
    p.add_argument("--n-components",     type=int, default=50)
    p.add_argument("--chunk-size",       type=int, default=5000)
    p.add_argument("--fit-out",          required=True, help="Output CSV for reference cohort")
    p.add_argument("--transform-out",    required=True, help="Output CSV for projected cohort")
    p.add_argument("--path-to-metadata-files",   required=True, help="Path to DemographicData.tsv and ProcessedSIRE_V7.tsv") 

    main(p.parse_args())