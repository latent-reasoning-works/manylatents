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
data_root=${dir_root}/'data/HGDP'
path_to_admix='/lustre06/project/6005588/shared/bin/admixture_linux-1.3.0'
genotypes='gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.match1000G_GSAs_dietnet'
raw_data_root='/lustre06/project/6065672/grenier2/DietNet/Generalisation/datasets_112023/HGDP_1KGP'
infile=${raw_data_root}/${genotypes} #plink without extension 
#labels=${raw_data_root}/'gnomad.genomes.v3.1.2.hgdp_1kg_subset_sample_meta.reduced.tsv'
labels='data/HGDP/HGDP_labels'
output_csv="${data_root}/fis_scores.csv"
tmp_dir="${data_root}/tmp"
#metadata_path='/lustre06/project/6065672/shared/DietNet/1KGB_POP24/HGDP_sub/labels.tsv'

# Load virtual environment
module load StdEnv/2020
module load plink/1.9b_6.21-x86_64
source /lustre06/project/6065672/sciclun4/Envs/phate_env/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${dir_root}/src:${PYTHONPATH}"

# Create tmp directory
mkdir -p $tmp_dir

###
### 2) Create Fis scores from plink files
###

echo 'Loading population labels'

# Extract unique populations
populations=$(tail -n +2 $labels | cut -d',' -f2 | sort | uniq)

# Generate keep files for each population
while IFS=, read -r sample population superpopulation; do
    if [[ "$sample" != "" && "$population" != "" && "$sample" != "population" ]]; then
        echo "$sample $sample" >> "${tmp_dir}/${population}_keep.txt"
    fi
done < $labels

# Initialize the output CSV file
echo "sample_id,population,fis" > $output_csv

# Loop through each population and compute Fis
for pop in $populations; do
    keep_file="${tmp_dir}/${pop}_keep.txt"
    plink --bfile $infile --keep $keep_file --het --out "${tmp_dir}/het_output_${pop}"
    
    # Parse the Fis scores and append to the CSV
    awk -v pop="$pop" 'NR > 1 {print $1 "," pop "," $6}' "${tmp_dir}/het_output_${pop}.het" >> $output_csv
    echo "Generated Fis for ${pop}"
done

echo "Fis scores merged into ${output_csv}"
