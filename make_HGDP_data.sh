#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --account=ctb-hussinju
#SBATCH --time=1:00:00 
#SBATCH --mem=124GB
#SBATCH --job-name=make_HGDP
#SBATCH --output=slurm_files/make_HGDP_%j.out
#SBATCH --error=slurm_files/make_HGDP_%j.err


###
### 1) Create environment
###


# Define paths
dir_root='/lustre06/project/6065672/sciclun4/ActiveProjects/phate_genetics'
data_root=${dir_root}/'data/HGDP'
path_to_admix='/lustre06/project/6005588/shared/bin/admixture_linux-1.3.0'
genotypes='gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.match1000G_GSAs_dietnet'
raw_data_root='/lustre06/project/6065672/grenier2/DietNet/Generalisation/datasets_112023/HGDP_1KGP'
infile=${raw_data_root}/${genotypes} #plink without extension 
basename=$data_root/${genotypes} #outputfile 
labels=${raw_data_root}/'gnomad.genomes.v3.1.2.hgdp_1kg_subset_sample_meta.reduced.tsv'

data_path_HGDP=${basename}'.hdf5'
unrelated_sampleid_1000G_path=$raw_data_root/'1000G_unrelated_samples_set_2504.txt'
#metadata_path=$raw_data_root/'gnomad.genomes.v3.1.2.hgdp_1kg_subset_sample_meta.reduced.tsv'
metadata_path='/lustre06/project/6065672/shared/DietNet/1KGB_POP24/HGDP_sub/labels.tsv'

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
interested_labels=("ASW" "ACB" "PUR" "CLM" "PEL" "MXL" "Colombian" "Surui" "Maya" "Karitiana" "Pima")

# Convert array to awk-readable string of patterns
label_patterns=$(printf "|%s" "${interested_labels[@]}")
label_patterns=${label_patterns:1}  # remove the first '|'

# Use awk to filter out the sample names with the specified labels
awk -v labels="$label_patterns" -F'\t' '$4 ~ labels {print $1}' "$labels" > $data_root/output_ids.txt


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

python ${dir_root}/src/create_HGDP_dataset.py --hdf5_data_file ${basename}.hdf5 --admix_sampleid_file ${data_root}/output_ids.txt --admix_ratios_file $data_root/AMR_ACB_ASW.3.Q --output_file_path ${dir_root}/data/HGDP --metadata_path ${metadata_path} --unrelated_sampleid_1000G_path ${unrelated_sampleid_1000G_path}
