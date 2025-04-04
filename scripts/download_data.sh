#!/bin/bash
# Downloads HGDP-1KGP zip file from Dropbox

# Dropbox direct download link
URL_HGDP_1KGP="https://www.dropbox.com/scl/fi/r9jbv7eg6xhqzl5jvopn0/HGDP-1KGP.tar.gz?rlkey=77iq45q0sxgwpb8zs0x8ixas6&st=lao3eyfx&dl=1"

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