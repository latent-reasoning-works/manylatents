#!/bin/bash

# Define path variables for reusability
SOFTWARE_DIR='/home/jupyter/packages/'
PROJECT_DIR='/home/jupyter/workspaces/phaterepresentationsforvisualizationofgeneticdata/V2'
DATA_DIR="$PROJECT_DIR/Data"
PCA_DIR="$DATA_DIR/Rye/PCA"
RYE_DIR="$DATA_DIR/Rye"
GOOGLE_PROJECT_ENV="$GOOGLE_PROJECT"

# Paths to relatedness files
RELATEDNESS_FILE="$DATA_DIR/Rye/relatedness.tsv"
FLAGGED_SAMPLES_FILE="$DATA_DIR/Rye/relatedness_flagged_samples.tsv"
EXCLUDE_SAMPLES_FILE="$DATA_DIR/Rye/exclude_samples.txt"

# Create necessary directories
mkdir -p $DATA_DIR $PCA_DIR $RYE_DIR

# Download relatedness files from Google Cloud Storage
if [ ! -f $RELATEDNESS_FILE ]; then
    gsutil -u $GOOGLE_PROJECT_ENV cp gs://fc-aou-datasets-controlled/v7/wgs/short_read/snpindel/aux/relatedness/relatedness.tsv $RELATEDNESS_FILE
else
    echo "$RELATEDNESS_FILE already exists, skipping download."
fi

if [ ! -f $FLAGGED_SAMPLES_FILE ]; then
    gsutil -u $GOOGLE_PROJECT_ENV cp gs://fc-aou-datasets-controlled/v7/wgs/short_read/snpindel/aux/relatedness/relatedness_flagged_samples.tsv $FLAGGED_SAMPLES_FILE
else
    echo "$FLAGGED_SAMPLES_FILE already exists, skipping download."
fi

# Check file sizes to verify they were downloaded correctly
wc $RELATEDNESS_FILE
wc $FLAGGED_SAMPLES_FILE

# Python step: Create exclude_samples.txt from relatedness_flagged_samples.tsv
if [ ! -f $EXCLUDE_SAMPLES_FILE ]; then
    python3 << EOF
import pandas as pd

# Load the relatedness flagged samples
input_file = "$FLAGGED_SAMPLES_FILE"
output_file = "$EXCLUDE_SAMPLES_FILE"

# Read the TSV file and process it to create FID and IID columns
df = pd.read_csv(input_file, sep="\t")
df['FID'] = df['sample_id']
df['IID'] = df['sample_id']

# Save the processed FID/IID data to exclude_samples.txt
df[['FID', 'IID']].to_csv(output_file, sep=" ", index=False, header=False)
print(f"Output saved to {output_file}")
EOF
else
    echo "$EXCLUDE_SAMPLES_FILE already exists, skipping creation."
fi

# Perform PCA using PLINK2
PCA_OUTPUT_PREFIX="$PCA_DIR/extractedChrAllPruned.30"
if [ ! -f "${PCA_OUTPUT_PREFIX}.eigenvec" ]; then
    echo "Running PCA with PLINK2..."
    plink2 --bfile "$DATA_DIR/1KGPHGDPAOU_V7/extractedChrAllPruned" \
           --remove $EXCLUDE_SAMPLES_FILE \
           --pca 30 approx \
           --out $PCA_OUTPUT_PREFIX \
           --threads 32 \
           --memory 96000
else
    echo "PCA already completed, skipping."
fi

# Create the population-to-group mapping file (Pop2Group.txt)
POP2GROUP_FILE="$DATA_DIR/Pop2Group.txt"
if [ ! -f $POP2GROUP_FILE ]; then
    cat <<EOT > $POP2GROUP_FILE
Pop	Subgroup	Group
forReferenceGIH	SouthIndian	SouthIndian
forReferenceSTU	SouthIndian	SouthIndian
forReferenceITU	SouthIndian	SouthIndian
forReferenceCHB	EastAsian	EastAsian
forReferenceDai	EastAsian	EastAsian
forReferenceJPT	EastAsian	EastAsian
forReferenceKHV	EastAsian	EastAsian
forReferenceShe	EastAsian	EastAsian
forReferenceTujia	EastAsian	EastAsian
forReferenceBedouin	WestAsian	WestAsian
forReferenceDruze	WestAsian	WestAsian
forReferencePalestinian	WestAsian	WestAsian
forReferenceBrahui	NorthIndian	NorthIndian
forReferenceMakrani	NorthIndian	NorthIndian
forReferenceBalochi	NorthIndian	NorthIndian
forReferenceFIN	NorthernEuropean	NorthernEuropean
forReferenceGBR	NorthWesternEuropean	NorthWesternEuropean
forReferenceIBS	IberianEuropean	IberianEuropean
forReferenceTSI	ItalianEuropean	ItalianEuropean
forReferenceTuscan	ItalianEuropean	ItalianEuropean
forReferenceESN	WesternAfrican	WesternAfrican
forReferenceGWD	SeneGambianAfrican	SeneGambianAfrican
forReferenceLWK	EastAfrican	EastAfrican
forReferenceMSL	SeneGambianAfrican	SeneGambianAfrican
forReferenceYRI	WesternAfrican	WesternAfrican
forReferenceBougainville	Oceania	Oceania
forReferencePapuanHighlands	Oceania	Oceania
forReferencePapuanSepik	Oceania	Oceania
forReferenceColombian	NativeAmerican	NativeAmerican
forReferenceKaritiana	NativeAmerican	NativeAmerican
forReferenceMaya	NativeAmerican	NativeAmerican
forReferencePEL	NativeAmerican	NativeAmerican
forReferencePima	NativeAmerican	NativeAmerican
forReferenceSurui	NativeAmerican	NativeAmerican
EOT
else
    echo "$POP2GROUP_FILE already exists, skipping creation."
fi

# Run Rye analysis
RYE_OUTPUT_PREFIX="$RYE_DIR/extractedChrAllPruned.30"
if [ ! -f "${RYE_OUTPUT_PREFIX}.clusters" ]; then
    echo "Running Rye analysis..."
    $SOFTWARE_DIR/rye/rye.R --eigenvec $PCA_OUTPUT_PREFIX.eigenvec \
                            --eigenval $PCA_OUTPUT_PREFIX.eigenval \
                            --pop2group $POP2GROUP_FILE \
                            --attempts 5 \
                            --threads 32 \
                            --iter 100 \
                            --pcs 30 \
                            --rounds 200 \
                            --output $RYE_OUTPUT_PREFIX
else
    echo "Rye analysis already completed, skipping."
fi

echo "Process complete."