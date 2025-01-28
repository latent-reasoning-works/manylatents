#!/bin/bash

# Reproducible Dataset Construction

# Define root path for the experiment
exp_root="path/to/manifold_genetics"
data_folder="${exp_root}/data"

# Load environment
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "Error: Virtual environment not found at .venv/bin/activate"
  exit 1
fi

# Create data directory if it doesn't exist
mkdir -p "${data_folder}"

# Download required files
gsutil cp gs://gcp-public-data--gnomad/release/3.1/secondary_analyses/hgdp_1kg_v2/metadata_and_qc/gnomad_meta_updated.tsv "${data_folder}/"
gsutil cp gs://gcp-public-data--gnomad/release/3.1/secondary_analyses/hgdp_1kg_v2/pca/pca_outliers.txt "${data_folder}/"
gsutil cp -r gs://gcp-public-data--gnomad/release/3.1/secondary_analyses/hgdp_1kg_v2/pca_preprocessing/related_sample_ids.ht "${data_folder}/"

# Process metadata
python "${exp_root}/scripts/make_metadata_file.py" --data_root "${data_folder}"
