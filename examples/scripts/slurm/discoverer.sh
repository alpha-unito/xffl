#!/bin/bash

#SBATCH --job-name=llama
#SBATCH --error=llama
#SBATCH --output=llama
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --account=ehpc-ben-2025b08-058
#SBATCH --partition=common
#SBATCH --qos=ehpc-ben-2025b08-058
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --exclusive

module load python/3.10/gcc/base/3.10.17
srun bash -c "{{ streamflow_command }}" 2>&1
