#!/bin/bash

#SBATCH --job-name=llama
#SBATCH --account=IscrC_HOPE_0
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --mem=0
#SBATCH --error=llama.err
#SBATCH --output=llama.out
#SBATCH --exclusive


# Activate env
source /leonardo/home/userexternal/gmittone/23_llama_sc24/facility/xffl/bin/activate

# Set env vars
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
export HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export N_NODE=$SLURM_NNODES
export NPROC_PER_NODE=$SLURM_GPUS_PER_NODE

echo "N_NODE: $N_NODE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "HEAD_NODE_IP: $HEAD_NODE_IP"

{{ streamflow_environment }}

cd {{ streamflow_workdir }}

# srun is for launching a torchrun process on each node, then torchrun will take care of the rest
# --nnodes is the number of nodes (thus several torchrun instances)
# --nproc_per_node is the number of GPUs on each node
# --rdzv_id must be an EQUAL number for all processes participating in training. 

# Example command executed
# srun torchrun --nnodes 2 --nproc_per_node 4 --rdzv_id 1234 --rdzv_backend c10d --rdzv_endpoint lrdn0081:29500 finetuning.py --enable_fsdp --model_name /leonardo_scratch/fast/IscrC_HOPE_0/llama/llama-2-7b/ --output_dir /leonardo_scratch/fast/IscrC_HOPE_0/llama/llama-2-7b-fine-tuned/ --dataset gsarti_clean_mc4_it_dataset --dataset_path /leonardo/home/userexternal/gmittone/workspace/23_llama_sc24/dataset --num_epochs 1 --num_workers_dataloader 4 --seed $RANDOM --tokenizer /leonardo_scratch/fast/IscrC_HOPE_0/llama/tokenizer/

srun sh -c "{{ streamflow_command }}"