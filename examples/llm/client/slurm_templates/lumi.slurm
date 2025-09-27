#!/bin/bash

#SBATCH --job-name=llama
#SBATCH --error=llama
#SBATCH --output=llama
#SBATCH --time=01:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --account=project_465001007
#SBATCH --partition=ju-standard-g
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --exclusive

echo "Loading modules..."
module load LUMI/23.09
module load partition/G
module load rocm/5.6.1

echo "Setting OMP_NUM_THREADS to 8..."
export OMP_NUM_THREADS=8

echo "Launching jobs..."
srun bash -c "{{ streamflow_command }}" 2>&1
