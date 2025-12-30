#!/bin/bash -ex

#SBATCH --job-name=xFFL
#SBATCH --output=xFFL.out
#SBATCH --error=xFFL.err
#SBATCH --nodes=2
#SBATCH --time=00:15:00
#SBATCH --account=EUHPC_B27_058
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --ntasks-per-node=4			# GPUs x 1 node
#SBATCH --cpus-per-task=8			# cores x 1 node / GPUs x 1 node
#SBATCH --gpus-per-node=4
#SBATCH --exclusive

srun bash -c "{{ streamflow_command }}" 2>&1
