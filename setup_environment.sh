#!/bin/bash

# Create conda environment
conda create -n behavioral_seg python=3.9 -y

# Activate the environment
source activate behavioral_seg

# Install requirements
pip install -r requirements.txt

echo "Environment setup complete! To activate the environment, run:"
echo "conda activate behavioral_seg" 