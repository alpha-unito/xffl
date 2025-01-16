#!/bin/bash

#SBATCH --job-name=llama
#SBATCH --error=llama.err
#SBATCH --output=llama.out
#SBATCH --nodes=1
#SBATCH --ntasks=4					# Nodes x GPUs x 1 node
#SBATCH --time=00:15:00
#SBATCH --account=ehpc55
#SBATCH --partition=acc_ehpc
#SBATCH --ntasks-per-node=4			# GPUs x 1 node
#SBATCH --cpus-per-task=8			# cores x 1 node / GPUs x 1 node
#SBATCH --gpus-per-node=4
#SBATCH --exclusive

# Parameters
export ROOT_FOLDER=/gpfs/scratch/ehpc55/23_llama_sc24
export HPC=marenostrum

echo "Loading $HPC modules..."
module load singularity/4.1.5

echo "Running xFFL on $HPC with srun..."
srun bash ${ROOT_FOLDER}/worker/scripts/exec_llama.sh $HPC $@ 2>&1
