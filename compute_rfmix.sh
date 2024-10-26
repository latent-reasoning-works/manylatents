#!/bin/bash
#SBATCH --cpus-per-task=12
#SBATCH --account=ctb-hussinju
#SBATCH --time=4:00:00 
#SBATCH --mem=124GB
#SBATCH --job-name=pre_rfmix_phasing
#SBATCH --output=slurm_files/pre_rfmix_phasing_%A_%a.out
#SBATCH --error=slurm_files/pre_rfmix_phasing_%A_%a.err
#SBATCH --array=1-22

###
### 1) Create environment
###

# Load necessary modules
module load StdEnv/2020
module load plink/1.9b_6.21-x86_64
module load tabix/0.2.6
module load bcftools
source /lustre06/project/6065672/sciclun4/Envs/phate_env/bin/activate

# Define paths
dir_root='/lustre06/project/6065672/sciclun4/ActiveProjects/phate_genetics'
data_root='/lustre06/project/6065672/shared/DietNet/1KGB_POP24/1KGP/WGS30X_raw'
out_root=${dir_root}/'data/1KGP_VCF'
genotypes='1000G.2504_WGS30x.GSA17k_MHI.intersectGSA.miss10perc.maf0.05.pruned.autosomes.noHLA.phased_imputed'

# Get chromosome number from the array job
CHR=22 #${SLURM_ARRAY_TASK_ID}   # Chromosomes 1 through 22

# Define filenames for chromosome-specific VCF and phasing output
basename=$data_root/${genotypes}  # Base PLINK file without extension
vcf_output=${out_root}/${genotypes}.chr${CHR}.vcf.gz
vcf_output_with_ac=${out_root}/${genotypes}.chr${CHR}.withAC.vcf.gz
phased_output=${out_root}/${genotypes}.chr${CHR}.phased.vcf.gz
bcf_output=${out_root}/${genotypes}.chr${CHR}.phased.bcf
map_file="/lustre06/project/6005588/shared/References/GRCh38/genetic_map/for_shapeit/with_chr/chr${CHR}.b38.gmap"

###
### RFMix Section
###

# Define paths for RFMix processing
#meta_file="/lustre06/project/6065672/shared/DietNet/1KGB_POP24/1KGP/labels_pop.tsv"
output_dir=${out_root}  # Same output directory
RFMIX_EXEC="/lustre06/project/6005588/shared/bin/RFMix2/rfmix/rfmix"
threads=8  # Adjust the number of threads

# Output file for filtered superpopulation labels
ref_samples_file="${output_dir}/reference_samples.txt"
query_samples_file="${output_dir}/query_samples.txt"
ref_samples_with_superpop_file="${output_dir}/reference_samples_with_superpop.txt"

# Use the phased VCF as input for RFMix
input_vcf="${phased_output}"
ref_vcf="${output_dir}/1000G_rfmix_chr${CHR}_ref.vcf.gz"
query_vcf="${output_dir}/1000G_rfmix_chr${CHR}_query.vcf.gz"
genetic_map="/lustre06/project/6005588/shared/References/GRCh38/genetic_map/plink.chr${CHR}.GRCh38.txt"

# Extract reference samples from the VCF
echo "Extracting reference samples for chromosome ${CHR}..."
bcftools view -S $ref_samples_file -Oz -o $ref_vcf $input_vcf
tabix -p vcf $ref_vcf

# Extract query samples (non-reference) from the VCF
echo "Extracting query samples for chromosome ${CHR}..."
bcftools view -S $query_samples_file -Oz -o $query_vcf $input_vcf
tabix -p vcf $query_vcf

# Run RFMix
output_prefix="${output_dir}/rfmix_output_chr${CHR}"
echo "Running RFMix for chromosome ${CHR}..."
$RFMIX_EXEC \
    -f $query_vcf \
    -r $ref_vcf \
    -m $ref_samples_with_superpop_file \
    -g $genetic_map \
    -o $output_prefix \
    --chromosome=${CHR} \
    --n-threads=$threads

# Check if RFMix ran successfully
if [ $? -eq 0 ]; then
    echo "RFMix completed successfully for chromosome ${CHR}."
else
    echo "Error running RFMix for chromosome ${CHR}!" >&2
    exit 1
fi

echo "Phasing and RFMix processing completed for chromosome ${CHR}."