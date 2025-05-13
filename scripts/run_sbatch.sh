#!/bin/bash
#SBATCH --account=ctb-hussinju
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=run_sweep
#SBATCH --output=outputs/slurm_outputs/run_sweep_%j.out
#SBATCH --error=outputs/slurm_outputs/run_sweep_%j.err

# Load modules if needed (e.g. python/3.9)
# module load python/3.9  # uncomment if your cluster requires it

# Activate your virtual environment
source /lustre06/project/6065672/sciclun4/ActiveProjects/ManyLatents/.venv/bin/activate

# Capture everything passed to this script
CMD="$@"

echo "Running command: $CMD"
eval "$CMD"
