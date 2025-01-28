#!/bin/bash

# Reproducible Dataset construction

# Load environment
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "Error: Virtual environment not found at .venv/bin/activate"
  exit 1
fi

# Download required files
mkdir -p  data
gsutil cp gs://gcp-public-data--gnomad/release/3.1/secondary_analyses/hgdp_1kg_v2/metadata_and_qc/gnomad_meta_updated.tsv data/
gsutil cp gs://gcp-public-data--gnomad/release/3.1/secondary_analyses/hgdp_1kg_v2/pca/pca_outliers.txt data/
gsutil cp -r gs://gcp-public-data--gnomad/release/3.1/secondary_analyses/hgdp_1kg_v2/pca_preprocessing/related_sample_ids.ht data/

# Process metadata
python scripts/make_metadata_file.py