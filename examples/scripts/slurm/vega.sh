#!/bin/bash

#SBATCH --job-name=xFFL
#SBATCH --output=xFFL.out
#SBATCH --error=xFFL.err
#SBATCH --nodes=2
#SBATCH --time=00:15:00
#SBATCH --account=b2024b11-066-users
#SBATCH --partition=gpu
##SBATCH --qos=test
#SBATCH --ntasks-per-node=4			# GPUs x 1 node
#SBATCH --cpus-per-task=4			# cores x 1 node / GPUs x 1 node
#SBATCH --gpus-per-node=4
#SBATCH --exclusive

srun bash -c "{{ streamflow_command }}" 2>&1
