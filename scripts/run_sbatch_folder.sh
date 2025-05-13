#!/bin/bash

# Example usage:
# bash scripts/run_sbatch_folder.sh sweeps/dlatree_default_mbyl

SWEEP_DIR=$1

if [ -z "src/configs/experiment/$SWEEP_DIR" ]; then
  echo "Usage: $0 <sweep_directory>"
  exit 1
fi

# Make sure the directory exists
if [ ! -d "src/configs/experiment/$SWEEP_DIR" ]; then
  echo "Directory $SWEEP_DIR does not exist!"
  exit 1
fi

# Loop over each YAML file in the sweep directory
for sweep_cfg in "src/configs/experiment/$SWEEP_DIR"/*.yaml; do
  # Extract the base name without extension
  cfg_name=$(basename "$sweep_cfg" .yaml)

  echo "Submitting: $cfg_name"
  sbatch scripts/run_sbatch.sh WANDB_MODE=offline python -m src.main experiment=${SWEEP_DIR}/${cfg_name}
done