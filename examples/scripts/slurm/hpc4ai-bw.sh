#!/bin/bash

#SBATCH --job-name=xFFL
#SBATCH --error=xFFL.err
#SBATCH --output=xFFL.out
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --time=00:15:00
#SBATCH --account=gmittone
#SBATCH --partition=broadwell
#SBATCH --gpus-per-node=1
#SBATCH --tasks-per-node=1
#SBATCH --exclusive

export GLOO_SOCKET_IFNAME=ibs1

#spack load /a4nuuh6
#spack load /bl3nmno # Apptainer

srun bash -c "{{ streamflow_command }}" 2>&1
