#!/bin/bash

#SBATCH --job-name=llama
#SBATCH --error=llama.err
#SBATCH --output=llama.out
#SBATCH --nodes=1
#SBATCH --ntasks=4					# Nodes x GPUs x 1 node
#SBATCH --time=00:05:00
#SBATCH --account=EUHPC_B18_066
#SBATCH --partition=boost_usr_prod
#SBATCH --ntasks-per-node=4			# GPUs x 1 node
#SBATCH --cpus-per-task=8			# cores x 1 node / GPUs x 1 node
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --qos=boost_qos_dbg

# Parameters
# export CODE_FOLDER=""
# export MODEL_FOLDER=""
# export DATASET_FOLDER=""
# export IMAGE=""
# export FACILITY="leonardo"

# Load modules
echo "Loading modules..."

# Execute
echo "Running xFFL on ${FACILITY} with srun..."
srun bash -c "{{ streamflow_command }}" 2>&1
