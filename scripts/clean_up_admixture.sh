#!/bin/bash
# Downloads HGDP-1KGP zip file from Dropbox

# Dropbox direct download link
root="data/HGDP+1KGP"

python scripts/clean_up_admixture.py --indices_file ${root}/"admixture/indices.txt" --metadata_path ${root}/"gnomad_derived_metadata_with_filtered_sampleids.csv" --admix_file_root ${root}/"admixture"
