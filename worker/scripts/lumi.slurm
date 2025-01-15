#!/bin/bash

#SBATCH --job-name=llama
#SBATCH --error=llama.err
#SBATCH --output=llama.out
#SBATCH --nodes=1
#SBATCH --ntasks=8					# Nodes x GPUs x 1 node
#SBATCH --time=00:15:00
#SBATCH --account=project_465001371
#SBATCH --partition=standard-g
#SBATCH --ntasks-per-node=8			# GPUs x 1 node
#SBATCH --cpus-per-task=7 #LUMI suggest 7 for a "clean execution"	# cores x 1 node / GPUs x 1 node
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --exclusive

# Parameters
export ROOT_FOLDER=/flash/project_465001371/23_llama_sc24
export HPC=lumi

echo "Loading $HPC modules..."
module load LUMI/24.03 partition/G PyTorch/2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617

c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

echo "Running xFFL on $HPC with srun..."
srun --cpu-bind=mask_cpu:$MYMASKS bash ${ROOT_FOLDER}/worker/scripts/exec_llama.sh $HPC $@ 2>&1
