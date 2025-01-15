#!/bin/bash
#SBATCH --job-name=llama
#SBATCH --error=llama.err
#SBATCH --output=llama.out
#SBATCH --nodes=8
#SBATCH --ntasks=32					# Nodes x GPUs x 1 node
#SBATCH --time=00:30:00
#SBATCH --account=IscrC_HOPE_0
#SBATCH --partition=boost_usr_prod
#SBATCH --ntasks-per-node=4			# GPUs x 1 node
#SBATCH --cpus-per-task=8			# cores x 1 node / GPUs x 1 node
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
# #SBATCH --qos=boost_qos_bprod		# For more than 128 nodes

echo "Loading modules..."

module load cuda/12.1
module load openmpi/4.1.4--nvhpc--23.1-cuda-11.8

echo "Running xFFL with srun..."
srun bash -c "{{ streamflow_command }}" 2>&1
