#!/bin/bash -l

#SBATCH --job-name=llama
#SBATCH --error=llama.err
#SBATCH --output=llama.out
#SBATCH --nodes=16
#SBATCH --ntasks=64				# Nodes x GPUs x 1 node
#SBATCH --time=00:30:00
#SBATCH --account=p200342
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=4		# GPUs x 1 node
#SBATCH --cpus-per-task=16		# cores x 1 node / GPUs x 1 node
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --qos=default


module load Apptainer/1.2.4-GCCcore-12.3.0

echo "Running xFFL with srun..."
srun bash -c "{{ streamflow_command }}" 2>&1
