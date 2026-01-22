#!/bin/bash

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

ulimit -n 131072
module load cuda/12.2 nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22

export PYTHONUNBUFFERED=1

# export NCCL_DEBUG=INFO
# export TORCH_NCCL_TRACE_BUFFER_SIZE=1000
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=SYS
export NCCL_NET_GDR_LEVEL=2

srun bash -c "{{ streamflow_command }}" 2>&1
