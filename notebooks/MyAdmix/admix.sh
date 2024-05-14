#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --account=ctb-hussinju
#SBATCH --time=4:00:00 
#SBATCH --mem=64GB
#SBATCH --job-name=run_admix
#SBATCH --output=../../slurm_files/run_admix_%j.out
#SBATCH --error=../../slurm_files/run_admix_%j.err

module load StdEnv/2020
module load plink/1.9b_6.21-x86_64

data_path=/lustre06/project/6065672/shared/DietNet/1KGB_POP24/1KGP
path_to_admix=/lustre06/project/6005588/shared/bin/admixture_linux-1.3.0
genotypes=WGS30X_raw/1000G.2504_WGS30x.GSA17k_MHI.intersectGSA.miss10perc.maf0.05.pruned.autosomes.noHLA.phased_imputed
labels=labels_pop_merged.tsv
output_path=/lustre06/project/6065672/sciclun4/ActiveProjects/phate_genetics/notebooks/MyAdmix

# A space-separated list of labels you're interested in
# For example, labels are "label1", "label2", "label3"
interested_labels=("ASW" "ACB" "MXL" "PEL" "PUR" "CLM")

# Convert array to awk-readable string of patterns
label_patterns=$(printf "|%s" "${interested_labels[@]}")
label_patterns=${label_patterns:1}  # remove the first '|'

# Use awk to filter out the sample names with the specified labels
awk -v labels="$label_patterns" -F'\t' '$2 ~ labels {print $1}' "$data_path/$labels" > $output_path/output_ids.txt

grep -Fwf $output_path/output_ids.txt $data_path/$genotypes'.fam' > $output_path/output_info.txt

plink --bfile $data_path/$genotypes --keep $output_path/output_info.txt --real-ref-alleles --make-bed --out AMR_ACB_ASW

$path_to_admix/admixture $output_path/AMR_ACB_ASW'.bed' 3