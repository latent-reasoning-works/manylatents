#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --account=ctb-hussinju
#SBATCH --time=12:00:00 
#SBATCH --mem=124GB
#SBATCH --job-name=hyperparam_exp_hgdp
#SBATCH --output=slurm_files/hyperparam_exp_hgdp_%j.out
#SBATCH --error=slurm_files/hyperparam_exp_hgdp_%j.err

###
### 1) Create environment
###

# Load virtual environment
module load StdEnv/2020
module load plink/1.9b_6.21-x86_64
source /lustre06/project/6065672/sciclun4/Envs/phate_env/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${dir_root}/src:${PYTHONPATH}"

python hyperparameter_search.py --output_dir results \
                                --data_dir /lustre06/project/6065672/shared/MattDataSharing/1KGP+HGDP/V4 \
                                --admixture_dir /lustre06/project/6065672/shared/MattDataSharing/1KGP+HGDP/V4/admixture/ADMIXTURE_HGDP+1KGP
