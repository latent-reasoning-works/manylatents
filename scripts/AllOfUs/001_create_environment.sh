#!/bin/bash

# Set paths for software and project directories
PATH_TO_SOFTWARE='/home/jupyter/packages/'
PATH_TO_PROJECT='/home/jupyter/workspaces/phaterepresentationsforvisualizationofgeneticdata'

# Create software directory if it doesn't exist
mkdir -p $PATH_TO_SOFTWARE

# Function to check if a command was successful
check_command() {
    if [ $? -ne 0 ]; then
        echo "Error occurred during $1. Exiting."
        exit 1
    fi
}

# Install RFMix
echo "Installing RFMix..."
cd $PATH_TO_SOFTWARE

# Check if the rfmix directory already exists
if [ -d "${PATH_TO_SOFTWARE}/rfmix" ]; then
    echo "RFMix directory already exists. Skipping cloning..."
else
    git clone https://github.com/slowkoni/rfmix.git
    check_command "cloning RFMix repository"
fi

# Build RFMix
cd rfmix
autoreconf --force --install   # Creates the configure script and dependencies
check_command "autoreconf for RFMix"
./configure                    # Generates the Makefile
check_command "configuring RFMix"
make                           # Builds RFMix
check_command "building RFMix"

# Install SHAPEIT5 static binaries
echo "Installing SHAPEIT5..."
cd $PATH_TO_SOFTWARE

# Check if the shapeit5 directory already exists
if [ -d "${PATH_TO_SOFTWARE}/shapeit5" ]; then
    echo "SHAPEIT5 directory already exists. Skipping cloning..."
else
    git clone https://github.com/odelaneau/shapeit5.git
    check_command "cloning SHAPEIT5 repository"
fi

# Download SHAPEIT5 binaries
cd shapeit5/static_bins
if [ ! -f "ligate_static" ]; then
    wget https://github.com/odelaneau/shapeit5/releases/download/v5.1.1/ligate_static
    check_command "downloading ligate_static"
fi
if [ ! -f "phase_common_static" ]; then
    wget https://github.com/odelaneau/shapeit5/releases/download/v5.1.1/phase_common_static
    check_command "downloading phase_common_static"
fi
if [ ! -f "phase_rare_static" ]; then
    wget https://github.com/odelaneau/shapeit5/releases/download/v5.1.1/phase_rare_static
    check_command "downloading phase_rare_static"
fi
if [ ! -f "simulate_static" ]; then
    wget https://github.com/odelaneau/shapeit5/releases/download/v5.1.1/simulate_static
    check_command "downloading simulate_static"
fi
if [ ! -f "switch_static" ]; then
    wget https://github.com/odelaneau/shapeit5/releases/download/v5.1.1/switch_static
    check_command "downloading switch_static"
fi
if [ ! -f "xcftools_static" ]; then
    wget https://github.com/odelaneau/shapeit5/releases/download/v5.1.1/xcftools_static
    check_command "downloading xcftools_static"
fi

# Make binaries executable
chmod +x ligate_static phase_common_static phase_rare_static simulate_static switch_static xcftools_static
check_command "making SHAPEIT5 binaries executable"


# Install Rye in the same directory as RFMix and SHAPEIT5
echo "Installing Rye in $PATH_TO_SOFTWARE..."
cd $PATH_TO_SOFTWARE

# Clone the Rye repository if it doesn't already exist
if [ ! -d "${PATH_TO_SOFTWARE}/rye" ]; then
    echo "Cloning Rye from the healthdisparities GitHub repository..."
    git clone -b dev https://github.com/healthdisparities/rye.git
    check_command "cloning Rye repository"
else
    echo "Rye repository already exists. Skipping cloning."
fi

# Install R dependencies for Rye
echo "Installing R dependencies for Rye..."
R -e 'install.packages(c("nnls", "optparse", "Hmisc"), repos = "https://cloud.r-project.org")'
check_command "installing R dependencies"

# Verify that Rye is correctly cloned and dependencies are installed
if [ -d "${PATH_TO_SOFTWARE}/rye" ]; then
    echo "Rye repository is set up successfully in $PATH_TO_SOFTWARE."
else
    echo "Error: Rye repository was not set up correctly."
    exit 1
fi

# Final message
echo "All software installed successfully!"