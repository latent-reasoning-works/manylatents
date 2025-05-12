#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

# Capture everything passed to this script
CMD="$@"

echo "Running command: $CMD"
eval "$CMD"
