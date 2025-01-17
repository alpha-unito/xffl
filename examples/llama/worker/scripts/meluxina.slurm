#!/bin/bash -l

#SBATCH --job-name=llama
#SBATCH --error=llama.err
#SBATCH --output=llama.out
#SBATCH --nodes=2
#SBATCH --ntasks=8				# Nodes x GPUs x 1 node
#SBATCH --time=00:15:00
#SBATCH --account=p200594
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=4		# GPUs x 1 node
#SBATCH --cpus-per-task=16		# cores x 1 node / GPUs x 1 node
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --qos=default

# Parameters
export ROOT_FOLDER=/project/home/p200594/23_llama_sc24
export HPC=meluxina

echo "Loading $HPC modules..."
module load Apptainer/1.2.4-GCCcore-12.3.0

echo "Running xFFL on $HPC with srun..."
srun bash ${ROOT_FOLDER}/worker/scripts/exec_llama.sh $HPC $@ 2>&1
