#!/bin/bash

# Set up conda directory
CONDA_DIR=/opt/conda

# Download and install Miniconda
wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
/bin/bash ~/miniconda.sh -b -p $CONDA_DIR
rm ~/miniconda.sh

# Add conda to path
export PATH=$CONDA_DIR/bin:$PATH

# Initialize conda
conda init bash

# Install mamba
conda install -y mamba -n base -c conda-forge

# Check environment.yml location and create conda environment
if [ -f "/tmp/environment.yml" ]; then
    ENV_FILE="/tmp/environment.yml"
else
    ENV_FILE="environment.yml"
fi

mamba env create -f $ENV_FILE

# Add conda environment auto-activation to .bashrc
echo "# Auto-activate conda environment" >> ~/.bashrc
echo "conda activate diffprivate" >> ~/.bashrc