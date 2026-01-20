#!/bin/bash

# Parameters
#SBATCH --array=0-4%5
#SBATCH --comment='LRW experiment'
#SBATCH --cpus-per-task=4
#SBATCH --error=/network/scratch/c/cesar.valdez/manyLatents/logs/ph_threshold_sweep/multiruns/2026-01-12/03-35-18/.submitit/%A_%a/%A_%a_0_log.err
#SBATCH --gpus-per-task=0
#SBATCH --job-name=main
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/network/scratch/c/cesar.valdez/manyLatents/logs/ph_threshold_sweep/multiruns/2026-01-12/03-35-18/.submitit/%A_%a/%A_%a_0_log.out
#SBATCH --partition=main
#SBATCH --signal=USR2@120
#SBATCH --time=240
#SBATCH --wckey=submitit

# setup
export WANDB_CACHE_DIR=$SCRATCH/.wandb_cache
export WANDB_CONFIG_DIR=$SCRATCH/.config/wandb
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Starting job on $(hostname)"
echo "SLURM_JOB_ID $SLURM_JOB_ID"

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /network/scratch/c/cesar.valdez/manyLatents/logs/ph_threshold_sweep/multiruns/2026-01-12/03-35-18/.submitit/%A_%a/%A_%a_%t_log.out --error /network/scratch/c/cesar.valdez/manyLatents/logs/ph_threshold_sweep/multiruns/2026-01-12/03-35-18/.submitit/%A_%a/%A_%a_%t_log.err /network/scratch/c/cesar.valdez/manyLatents/.venv/bin/python3 -u -m submitit.core._submit /network/scratch/c/cesar.valdez/manyLatents/logs/ph_threshold_sweep/multiruns/2026-01-12/03-35-18/.submitit/%j
