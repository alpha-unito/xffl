#!/bin/bash

#SBATCH --job-name=llama
#SBATCH --account=DD-23-165
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --error=llama.err
#SBATCH --output=llama.out


# Activate env
source /home/it4i-giuliom/src/23_llama_sc24/facility/xffl/bin/activate


# Set env vars
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export N_NODE=$SLURM_NNODES
export NPROC_PER_NODE=$SLURM_GPUS_PER_NODE
echo "N_NODE: $N_NODE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "HEAD_NODE_IP: $HEAD_NODE_IP"

{{ streamflow_environment }}

cd {{ streamflow_workdir }}

srun sh -c "{{ streamflow_command }}"

