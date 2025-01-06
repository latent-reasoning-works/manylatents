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
genotypes='1000G.2504_WGS30x.GSA17k_MHI.intersectGSA.miss10perc.maf0.05.pruned.autosomes.noHLA.phased_imputed'
infile='/lustre06/project/6065672/shared/DietNet/1KGB_POP24/1KGP/WGS30X_raw'/${genotypes} # plink without extension
labels='data/1000G/1000G_labels'
output_csv="${data_root}/fis_scores.csv"
tmp_dir="${data_root}/tmp"

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
