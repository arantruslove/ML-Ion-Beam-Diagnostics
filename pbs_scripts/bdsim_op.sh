#!/bin/bash
#PBS -l walltime=21:0:0
#PBS -l select=1:ncpus=100:mem=256gb
#PBS -N bdsim_op
#PBS -o logs/
#PBS -e logs/

# Navigate to the directory where the SIF file is located
cd $PBS_O_WORKDIR

# Run the Apptainer container and execute the subsequent commands
apptainer exec apptainer.sif /bin/bash -c "
  source /root/.bashrc
  cd src
  python3 bdsim_op.py > ../logs/bdsim_op.log 2>&1
"