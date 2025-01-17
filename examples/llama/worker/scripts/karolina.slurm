#!/usr/bin/bash

#SBATCH --job-name=llama
#SBATCH --error=llama.err
#SBATCH --output=llama.out
#SBATCH --nodes=1
#SBATCH --ntasks=8					# Nodes x GPUs x 1 node
#SBATCH --time=00:15:00
#SBATCH --account=DD-24-76
#SBATCH --partition=qgpu
#SBATCH --ntasks-per-node=8			# GPUs x 1 node
#SBATCH --cpus-per-task=16			# cores x 1 node / GPUs x 1 node
#SBATCH --gpus-per-node=8

# Parameters
export ROOT_FOLDER=/scratch/project/dd-24-76/23_llama_sc24
export HPC=karolina

echo "Running xFFL on $HPC with srun..."
srun bash ${ROOT_FOLDER}/worker/scripts/exec_llama.sh $HPC $@ 2>&1
