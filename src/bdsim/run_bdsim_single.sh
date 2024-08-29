#!/bin/bash
current_dir=$5
source /root/.bashrc
cd "$current_dir/bdsim_scripts"

# Variables for the job
PARTICLE_TYPE=$1
JOB_ID=$2
NUM_PARTICLES=$3
NUMBER=$4
bdsim --file=${PARTICLE_TYPE}beam_${JOB_ID}_${NUMBER}.gmad --outfile=../bdsim_outputs/${PARTICLE_TYPE}_output_${JOB_ID}_${NUMBER} --batch --ngenerate=${NUM_PARTICLES}