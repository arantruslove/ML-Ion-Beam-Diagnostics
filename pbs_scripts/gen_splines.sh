#!/bin/bash
#PBS -l walltime=1:0:0
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -N gen_splines
#PBS -o logs/
#PBS -e logs/


# Navigate to the working directory
cd $PBS_O_WORKDIR


# Run the Apptainer container and execute the subsequent commands
apptainer exec apptainer.sif /bin/bash -c "
  source /root/.bashrc
  cd src/custom/splines
  python3 create_proton_splines.py > ../../../logs/gen_proton_spline.log 2>&1
  python3 create_electron_splines.py > ../../../logs/gen_electron_spline.log 2>&1
"