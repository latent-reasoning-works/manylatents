#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=124GB
#SBATCH --job-name=run_admixture_HGDP
#SBATCH --output=${dir_root}/slurm_files/make_HGDP_admixture_%j.out
#SBATCH --error=${dir_root}/slurm_files/make_HGDP_admixture_%j.err

###
### 1) Define paths and set up environment
###

# Define paths
dir_root="/path/to/manifold_genetics"
metadata_path="${dir_root}/data/gnomad_derived_metadata_with_filtered_sampleids.csv"
admix_file_root="${dir_root}/data/admixture/ADMIXTURE_HGDP+1KGP"
path_to_admix="/path/to/admixture"
genotypes="gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.match1000G_GSAs_dietnet"
infile="${dir_root}/data/genotypes/${genotypes}"

# Activate virtual environment
source "${dir_root}/.venv/bin/activate"

###
### 2) Prepare metadata (subsets)
###

python "${dir_root}/scripts/make_subset_for_admixture.py" --metadata_path "${metadata_path}" --admix_file_root "${admix_file_root}"

###
### 3) Prepare plink subsets
###

# Make bed files for subsets
/path/to/plink --bfile "${infile}" --keep "${admix_file_root}/tmp/AMR_ACB_ASW_samples_to_keep" --real-ref-alleles --make-bed --out "${admix_file_root}/tmp/AMR_ACB_ASW"
/path/to/plink --bfile "${infile}" --keep "${admix_file_root}/tmp/AMR_EUR_AFR_samples_to_keep" --real-ref-alleles --make-bed --out "${admix_file_root}/tmp/AMR_EUR_AFR"
/path/to/plink --bfile "${infile}" --keep "${admix_file_root}/tmp/AMR_ACB_ASW_1KGP_ONLY_samples_to_keep" --real-ref-alleles --make-bed --out "${admix_file_root}/tmp/AMR_ACB_ASW_1KGP_ONLY"
/path/to/plink --bfile "${infile}" --keep "${admix_file_root}/tmp/global_samples_to_keep" --real-ref-alleles --make-bed --out "${admix_file_root}/tmp/global"

###
### 4) Compute admixture ratios from plink files
###

# Admixture global
for k in {2..9}; do
    "${path_to_admix}/admixture" "${admix_file_root}/tmp/global.bed" $k -j16
done

# Admixture AMR+ACB+ASW
for k in {2..6}; do
    "${path_to_admix}/admixture" "${admix_file_root}/tmp/AMR_ACB_ASW.bed" $k -j16
done

# Admixture AMR_EUR_AFR
for k in {2..6}; do
    "${path_to_admix}/admixture" "${admix_file_root}/tmp/AMR_EUR_AFR.bed" $k -j16
done

###
### 5) Export admixture to csv files
###

python "${dir_root}/scripts/clean_up_admixture.py" --metadata_path "${metadata_path}" --admix_file_root "${admix_file_root}"