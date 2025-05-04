#!/bin/bash
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00

# Capture everything passed to this script
CMD="$@"

echo "Running command: $CMD"
eval "$CMD"
