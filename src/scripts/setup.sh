
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

# Create conda environment
mamba env create -f /tmp/environment.yaml