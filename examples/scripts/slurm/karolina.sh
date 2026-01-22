#!/bin/bash

#SBATCH --job-name=xFFL
#SBATCH --output=xFFL.out
#SBATCH --error=xFFL.err
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --account=EU-25-98
#SBATCH --partition=qgpu_exp
#SBATCH --ntasks-per-node=8			# GPUs x 1 node
##SBATCH --cpus-per-task=8			# cores x 1 node / GPUs x 1 node
#SBATCH --gpus-per-node=8
#SBATCH --exclusive

export APPTAINER_BINDPATH=$SINGULARITY_BINDPATH
export APPTAINERENV_LD_PRELOAD=$SINGULARITYENV_LD_PRELOAD

srun bash -c "{{ streamflow_command }}" 2>&1
