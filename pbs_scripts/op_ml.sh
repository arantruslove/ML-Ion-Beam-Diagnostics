#!/bin/bash
#PBS -l walltime=23:0:0
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=RTX6000
#PBS -N op_ml
#PBS -o logs/
#PBS -e logs/

module purge
module add tools/dev
module add TensorFlow/2.11.0-foss-2022a-CUDA-11.8.0

# Create the venv directory if it doesn't exist
mkdir -p ~/venv

# Create the virtual environment if it doesn't already exist
if [ ! -d "~/venv/example-env" ]; then
    python3 -m venv ~/venv/example-env
fi

# Activate the virtual environment
source ~/venv/example-env/bin/activate

# Upgrade pip to the latest version within the virtual environment
python3 -m pip install --upgrade pip

# Navigate to the working directory (where your Python script is located)
cd $PBS_O_WORKDIR

# Install custom Python modules using pip
python3 -m pip install -r dependencies/requirements_optimise.txt

# Execute the Python script
python3 op_ml.py

# Deactivate the virtual environment
deactivate

# Deleting the virtual environment (optional, consider keeping it for re-use)
rm -rf ~/venv/example-env
