#!/bin/bash
#PBS -l walltime=3:0:0
#PBS -l select=1:ncpus=64:mem=128gb
#PBS -N custom_generate
#PBS -o logs/
#PBS -e logs/

# Navigate to the directory where the SIF file is located
cd $PBS_O_WORKDIR

# Run the Apptainer container and execute the subsequent commands
apptainer exec apptainer.sif /bin/bash -c "
  source /root/.bashrc
  cd src
  python3 custom_generate.py > ../logs/custom_generate.log 2>&1
"