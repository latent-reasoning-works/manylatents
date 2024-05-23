#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --account=ctb-hussinju
#SBATCH --time=6:00:00 
#SBATCH --mem=64GB
#SBATCH --job-name=phate_hyperparam_check
#SBATCH --output=slurm_files/phate_hyperparam_check_%j.out
#SBATCH --error=slurm_files/phate_hyperparam_check_%j.err

source /lustre06/project/6065672/sciclun4/Envs/phate_env/bin/activate

# Define paths
python check_phate_hyperparams.py
