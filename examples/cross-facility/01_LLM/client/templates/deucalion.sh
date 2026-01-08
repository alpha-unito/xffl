#!/bin/bash

#SBATCH --job-name=xFFL
#SBATCH --output=xFFL.out
#SBATCH --error=xFFL.err
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --account=eehpc-ben-2025b08-058g
#SBATCH --partition=dev-a100-80
##SBATCH --qos=test
#SBATCH --ntasks-per-node=4			# GPUs x 1 node
#SBATCH --cpus-per-task=4			# cores x 1 node / GPUs x 1 node
#SBATCH --gpus-per-node=4
#SBATCH --exclusive

module load Python/3.11.5-GCCcore-13.2.0

srun bash -c "{{ streamflow_command }}" 2>&1

#srun bash -c """
#export XFFL_LOCAL_TMPDIR=${LOCALSCRATCH}
#module load Python/3.11.5-GCCcore-13.2.0
#{{ streamflow_command }}
#""" 2>&1
