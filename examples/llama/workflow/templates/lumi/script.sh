#!/bin/bash

#SBATCH --job-name=llama
#SBATCH --error=llama.err
#SBATCH --output=llama.out
#SBATCH --nodes=8
#SBATCH --ntasks=64					# Nodes x GPUs x 1 node
#SBATCH --time=00:30:00
# #SBATCH --account=project_465000998
#SBATCH --account=project_465001007
#SBATCH --partition=standard-g
#SBATCH --ntasks-per-node=8			# GPUs x 1 node
#SBATCH --cpus-per-task=7			# cores x 1 node / GPUs x 1 node
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --exclusive


echo "Loading modules..."
# module load LUMI/23.09
# module load partition/G
# module load rocm/5.6.1
c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"


module load LUMI/23.09 PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240315

echo "Running xFFL with srun..."
srun bash -c "{{ streamflow_command }}" 2>&1
