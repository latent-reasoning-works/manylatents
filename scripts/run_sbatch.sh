#!/bin/bash
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=8:00:00

# Capture everything passed to this script
CMD="$@"

echo "Running command: $CMD"
eval "$CMD"
