#!/bin/bash
current_dir=$5
cd "$current_dir/bdsim_outputs"

PARTICLE_TYPE=$1
job_ID=$2
NUM_RUNS=$3
START=$4

for ((i = START; i <= NUM_RUNS; i++))
do
    rebdsimHistoMerge ${PARTICLE_TYPE}_output_${job_ID}_${i}.root ${PARTICLE_TYPE}_output_${job_ID}_${i}.root
done