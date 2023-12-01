#!/bin/bash

#SBATCH --job-name=flask_server
#SBATCH --output=flask_server.out
#SBATCH --error=flask_server.err
#SBATCH --time=01:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --exclusive 
#SBATCH --mem=0

# Load necessary modules
module load cuda/12.2.1
module load python/3.9.13

# Name of the virtual environment
VENV_NAME="env"

# Check if the virtual environment already exists
if [ ! -d "$VENV_NAME" ]
then
    # The virtual environment doesn't exist, create it
    python3.9 -m venv $VENV_NAME
fi

# Activate the virtual environment
source $VENV_NAME/bin/activate

# Install the dependencies
pip install -r requirements.txt

# nvidia-smi

# Run the Flask server
python server.py

# sleep inf