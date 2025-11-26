#!/bin/bash

#SBATCH --job-name=llama
#SBATCH --error=llama
#SBATCH --output=llama
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --account=project_465002202
#SBATCH --partition=dev-g
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --exclusive

srun bash -c "{{ streamflow_command }}" 2>&1
