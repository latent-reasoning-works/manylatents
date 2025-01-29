#!/bin/bash

# Reproducible Dataset Construction

script_dir="$(dirname "$(realpath "$0")")"
exp_root="$(realpath "$script_dir/..")"  # Moves up one directory to project root

echo "Detected project root: $exp_root"

data_folder="${exp_root}/data"

echo "Data folder: $data_folder"
echo "Script directory: ${exp_root}/scripts"

# Load environment
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "Error: Virtual environment not found at .venv/bin/activate"
  exit 1
fi

# Create data directory if it doesn't exist
mkdir -p "${data_folder}"

# Verify gsutil exists
if ! command -v gsutil &> /dev/null; then
  echo "Error: gsutil not found! Install Google Cloud SDK."
  exit 1
fi

# Download required files
gsutil cp gs://gcp-public-data--gnomad/release/3.1/secondary_analyses/hgdp_1kg_v2/metadata_and_qc/gnomad_meta_updated.tsv "${data_folder}/"
gsutil cp gs://gcp-public-data--gnomad/release/3.1/secondary_analyses/hgdp_1kg_v2/pca/pca_outliers.txt "${data_folder}/"

# We converted to csv to remove hail dependency
#gsutil cp -r gs://gcp-public-data--gnomad/release/3.1/secondary_analyses/hgdp_1kg_v2/pca_preprocessing/related_sample_ids.ht "${data_folder}/"

# Process metadata
python "${exp_root}/scripts/make_metadata_file.py" --data_root "${data_folder}"
