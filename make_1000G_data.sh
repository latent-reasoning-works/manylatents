#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --account=ctb-hussinju
#SBATCH --time=1:00:00 
#SBATCH --mem=124GB
#SBATCH --job-name=make_1000G
#SBATCH --output=slurm_files/make_1000G_%j.out
#SBATCH --error=slurm_files/make_1000G_%j.err


###
### 1) Create environment
###


# Define paths
dir_root='/lustre06/project/6065672/sciclun4/ActiveProjects/phate_genetics'
data_root=${dir_root}/'data/1000G'
path_to_admix='/lustre06/project/6005588/shared/bin/admixture_linux-1.3.0'
genotypes='1000G.2504_WGS30x.GSA17k_MHI.intersectGSA.miss10perc.maf0.05.pruned.autosomes.noHLA.phased_imputed'
infile='/lustre06/project/6065672/shared/DietNet/1KGB_POP24/1KGP/WGS30X_raw'/${genotypes} #plink without extension 
basename=$data_root/${genotypes} #outputfile 
labels='/lustre06/project/6065672/shared/DietNet/1KGB_POP24/1KGP/labels_pop_merged.tsv'

data_path_1000G=${basename}'.hdf5'

# Load virtual environment
module load StdEnv/2020
module load plink/1.9b_6.21-x86_64
source /lustre06/project/6065672/sciclun4/Envs/phate_env/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${dir_root}/src:${PYTHONPATH}"

###
### 2) Create h5 file from plink files
###

# Prepares raw plink files (will create h5 file out of it).
plink --bfile $infile --recode A --real-ref-alleles --out $basename
cut -d' ' -f1,7- $basename.raw | tr ' ' \\t > $basename.raw.cols1and7toend.tab 

###
### 3) Compute admixture ratios from plink files
###

# A space-separated list of labels you're interested in
# For example, labels are "label1", "label2", "label3"
interested_labels=("ASW" "ACB" "MXL" "PEL" "PUR" "CLM")

# Convert array to awk-readable string of patterns
label_patterns=$(printf "|%s" "${interested_labels[@]}")
label_patterns=${label_patterns:1}  # remove the first '|'

# Use awk to filter out the sample names with the specified labels
awk -v labels="$label_patterns" -F'\t' '$2 ~ labels {print $1}' "$labels" > $data_root/output_ids.txt

grep -Fwf $data_root/output_ids.txt ${infile}.fam > $data_root/output_info.txt

# make bed files of subset to run admixture on
plink --bfile $infile --keep $data_root/output_info.txt --real-ref-alleles --make-bed --out $data_root/AMR_ACB_ASW

# Computing admixture ratios
$path_to_admix/admixture ${data_root}/AMR_ACB_ASW.bed 3

# Move the output files to the desired directory
mv ${dir_root}/AMR_ACB_ASW.3.P ${data_root}/
mv ${dir_root}/AMR_ACB_ASW.3.Q ${data_root}/
mv ${dir_root}/AMR_ACB_ASW.3.log ${data_root}/

###
### 4) Convert h5 files into format used by our pipeline
###

# converting plink files to h5py (what we read into python)
python ${dir_root}/src/convert_to_h5py.py --exp-path $dir_root --genotypes $basename.raw.cols1and7toend.tab --class-labels $labels --ncpus 20 --out $basename 

python ${dir_root}/src/create_1000G_dataset.py --hdf5_data_file ${basename}.hdf5 --admix_sampleid_file ${data_root}/output_ids.txt --admix_ratios_file $data_root/AMR_ACB_ASW.3.Q --output_file_path ${dir_root}/data/1000G
