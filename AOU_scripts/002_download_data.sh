#!/bin/bash

# This script sets up the workspace, downloads necessary data, and processes it using PLINK.
# Using $GOOGLE_PROJECT for paths and storage

# Check if required environment variables are set
if [ -z "$GOOGLE_PROJECT" ]; then
  echo "Error: GOOGLE_PROJECT must be set."
  exit 1
fi

# Function to download data from Google Cloud Storage
download_data() {
    local bucket_path=$1
    local dest_path=$2
    local file_name=$3

    # Check if the file has already been downloaded
    if [ -f "$dest_path/$file_name" ]; then
        echo "$file_name already exists, skipping download..."
    else
        echo "Downloading $file_name from $bucket_path..."

        # Ensure destination path exists
        mkdir -p "$dest_path"

        # Download the file using gsutil, and check if the download was successful
        gsutil -u "$GOOGLE_PROJECT" cp "$bucket_path/$file_name" "$dest_path/$file_name"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to download $file_name from $bucket_path."
            exit 1
        fi
    fi
}

# Function to process the downloaded dataset (splits by chromosome)
process_data() {
    local base_path=$1
    local dataset_name=$2
    local dest_path=$3

    # Ensure PLINK is installed before continuing
    if ! command -v plink &> /dev/null; then
        echo "Error: PLINK is not installed or not in the system's PATH."
        exit 1
    fi

    # Check if all 22 .bed files exist
    missing_files=0
    for i in $(seq 1 22); do
        if [ ! -f "$dest_path/extractedChr$i.bed" ]; then
            missing_files=1
            break
        fi
    done

    if [ $missing_files -eq 0 ]; then
        echo "$dataset_name already processed for all chromosomes, skipping..."
    else
        # Split the dataset into 22 files, one for each chromosome, and run PLINK for each in parallel
        echo "Splitting $dataset_name by chromosome..."
        seq 1 22 | xargs -I {} -P 22 sh -c "plink --bfile $base_path/$dataset_name --keep-allele-order --allow-no-sex --chr {} --make-bed --out $dest_path/extractedChr{}"
    fi
}

# Set up directories for new data output in V2
echo "Setting up Data/V2 directories..."
mkdir -p Data/1KGPHGDP
mkdir -p Data/AllofUs_V7
mkdir -p Data/Metadata

# Download 1KGP+HGDP data
download_data "gs://fc-secure-47ccf5a8-b9ba-460a-aa03-dea8d260953b/Data" "Data/1KGPHGDP" "1KGPHGDP.tar.gz"

# Extract the tar.gz file for 1KGP+HGDP if not already extracted
if [ -d "Data/1KGPHGDP/extracted_files" ]; then
    echo "1KGP+HGDP data already extracted, skipping..."
else
    echo "Extracting 1KGPHGDP data..."
    tar -xvf "Data/1KGPHGDP/1KGPHGDP.tar.gz" --directory "Data/1KGPHGDP/"
    
    # Move the extracted files out of the subdirectory if needed
    if [ -d "Data/1KGPHGDP/1KGPHGDP" ]; then
        echo "Moving extracted files out of subdirectory..."
        mv Data/1KGPHGDP/1KGPHGDP/* Data/1KGPHGDP/
        rmdir Data/1KGPHGDP/1KGPHGDP
    fi

    mkdir -p "Data/1KGPHGDP/extracted_files" # Marker directory to indicate extraction completed
    if [ ! -f "Data/1KGPHGDP/extractedChrAllUnpruned.bed" ]; then
        echo "Extraction incomplete. Missing key files. Exiting..."
        exit 1
    fi
fi

# Add the prefix "chr" to the .bim file to match with how All of Us supplies the data, if not already done
# Add the prefix "chr" to the .bim file if it does not already have it
if [ -f "Data/1KGPHGDP/extractedChrAllUnpruned.Prefix.bim" ]; then
    echo ".bim file already modified with chr prefix, skipping..."
else
    echo "Modifying .bim file to include 'chr' prefix..."
    # Check if "chr" is already present in the first field, and add it only if it's not
    awk '{if ($1 !~ /^chr/) {print "chr"$1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6} else {print $0}}' "Data/1KGPHGDP/extractedChrAllUnpruned.bim" > "Data/1KGPHGDP/extractedChrAllUnpruned.Prefix.bim"

    # Backup the original .bim file and replace it with the new one
    echo "Backing up and replacing .bim file..."
    mv "Data/1KGPHGDP/extractedChrAllUnpruned.bim" "Data/1KGPHGDP/extractedChrAllUnpruned.Original.bim"
    mv "Data/1KGPHGDP/extractedChrAllUnpruned.Prefix.bim" "Data/1KGPHGDP/extractedChrAllUnpruned.bim"
fi

# Process 1KGP+HGDP dataset
process_data "Data/1KGPHGDP" "extractedChrAllUnpruned" "Data/1KGPHGDP"

# Download AoU data
download_data "gs://fc-aou-datasets-controlled/v7/microarray/plink_v7.1" "Data/AllofUs_V7" "arrays.bed"
download_data "gs://fc-aou-datasets-controlled/v7/microarray/plink_v7.1" "Data/AllofUs_V7" "arrays.bim"
download_data "gs://fc-aou-datasets-controlled/v7/microarray/plink_v7.1" "Data/AllofUs_V7" "arrays.fam"

# Fix the AoU FAM file if not already fixed
if [ -f "Data/AllofUs_V7/arrays.Original.fam" ]; then
    echo "FAM file already fixed, skipping..."
else
    echo "Fixing FAM file for AoU data..."
    mv "Data/AllofUs_V7/arrays.fam" "Data/AllofUs_V7/arrays.Original.fam"
    awk '{print "AOU\t"$2"\t"$3"\t"$4"\t"$5"\t-9"}' "Data/AllofUs_V7/arrays.Original.fam" > "Data/AllofUs_V7/arrays.fam"
fi

# Process AoU dataset
process_data "Data/AllofUs_V7" "arrays" "Data/AllofUs_V7"

# Run Python queries to extract demographic and condition data
if [ -f "Data/Metadata/DemographicData.tsv" ] && [ -f "Data/Metadata/AllDiseases.tsv" ]; then
    echo "Demographic and disease data already extracted, skipping..."
else
    echo "Running Python queries to extract demographic and condition data..."
    python3 run_queries.py
fi

# Run the R script for merging and harmonizing genotype data
if [ -f "Data/1KGPHGDPAOU_V7/extractedChrAllPruned.bed" ]; then
    echo "Merged and harmonized data already exists, skipping..."
else
    echo "Running R script to merge and harmonize genotype data..."
    mkdir -p Data/1KGPHGDPAOU_V7
    Rscript merge_and_harmonize_genotypes.R
fi

# Run the R script for demographic data processing
if [ -f "Data/Metadata/ProcessedSIRE_V7.tsv" ]; then
    echo "Demographic data already processed, skipping..."
else
    echo "Processing demographic data with R script..."
    Rscript CreateCohortTable.R "Data/Metadata/" "Data/1KGPHGDPAOU_V7/extractedChrAllPruned.fam"
fi

# Final message to indicate the process is complete
echo "Process complete."