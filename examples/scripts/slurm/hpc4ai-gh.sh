#!/bin/bash

#SBATCH --job-name=xFFL
#SBATCH --error=xFFL.err
#SBATCH --output=xFFL.out
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --time=00:15:00
#SBATCH --account=gmittone
#SBATCH --partition=gracehopper
#SBATCH --gpus-per-node=1
#SBATCH --tasks-per-node=1
#SBATCH --exclusive

cd /beegfs/home/gmittone/xffl
source .venv-gh/bin/activate
spack load python@3.12.1 cuda@12.3.2 cudnn@8.9.5.30-12

export PYTHONUNBUFFERED=1

# killall python > /dev/null 2>&1
GLOO_SOCKET_IFNAME=$(ip --brief link | awk '{print $1}' | grep -E 'enx.*' | tr -d '\n')
source /etc/profile > /dev/null 2>&1
spack load /w4pv4j6

srun bash -c "{{ streamflow_command }}" 2>&1
