#!/bin/bash -ex

#SBATCH --job-name=xFFL
#SBATCH --output=xFFL.out
#SBATCH --error=xFFL.err
#SBATCH --nodes=2
#SBATCH --time=00:15:00
#SBATCH --account=p201021
#SBATCH --partition=gpu
#SBATCH --qos=test
#SBATCH --ntasks-per-node=4			# GPUs x 1 node
#SBATCH --cpus-per-task=4			# cores x 1 node / GPUs x 1 node
#SBATCH --gpus-per-node=4
#SBATCH --exclusive

. ${HOME}/.bashrc
if [ ! command -v module > /dev/null ]; then
    echo "module not found"
    exit 1
fi

srun bash -c "{{ streamflow_command }}" 2>&1
