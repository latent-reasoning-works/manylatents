#!/bin/bash
# Downloads HGDP-1KGP zip file from Dropbox

# Dropbox direct download link
URL_HGDP_1KGP="https://www.dropbox.com/scl/fi/gmq9fzo8yr2qpvaxhe3et/HGDP-1KGP.tar.gz?rlkey=h3nqkbhnmtnl2vczpwqrz0bul&st=e5r325gq&dl=1"

# Local path for the zip file
FILE_HGDP_1KGP="data/HGDP+1KGP.tar.gz"

# Local extraction directory
DIR_HGDP_1KGP="data/HGDP+1KGP"

# Download the file if it doesn't already exist
if [ ! -f "$FILE_HGDP_1KGP" ]; then
    echo "Downloading HGDP+1KGP..."
    wget -O "$FILE_HGDP_1KGP" "$URL_HGDP_1KGP"
else
    echo "File $FILE_HGDP_1KGP already exists, skipping download."
fi

# Extract only if the directory doesn't exist
if [ ! -d "$DIR_HGDP_1KGP" ]; then
    echo "Extracting HGDP+1KGP.tar.gz..."
    tar -xzvf "$FILE_HGDP_1KGP" -C data/
else
    echo "Directory $DIR_HGDP_1KGP already exists, skipping extraction."
fi


# Downloads scRNAseq zip file from Dropbox

# Dropbox direct download link
URL_scRNAseq="https://www.dropbox.com/scl/fi/f512xl4b128q4t55jdsfc/scRNAseq.tar.gz?rlkey=d2rzidzm8asmvegkq7blo7che&st=xo8cxkt7&dl=1"

# Local path for the zip file
FILE_scRNAseq="data/scRNAseq.tar.gz"

# Local extraction directory
DIR_scRNAseq="data/scRNAseq"

# Download the file if it doesn't already exist
if [ ! -f "$FILE_scRNAseq" ]; then
    echo "Downloading scRNAseq..."
    wget -O "$FILE_scRNAseq" "$URL_scRNAseq"
else
    echo "File $FILE_scRNAseq already exists, skipping download."
fi

# Extract only if the directory doesn't exist
if [ ! -d "$DIR_scRNAseq" ]; then
    echo "Extracting scRNAseq.tar.gz..."
    tar -xzvf "$FILE_scRNAseq" -C data/
else
    echo "Directory $DIR_scRNAseq already exists, skipping extraction."
fi