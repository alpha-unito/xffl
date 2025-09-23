#!/bin/bash -ex

#SBATCH --chdir=/leonardo_scratch/fast/uToID_bench/xffl/examples/simulation/03_LLM/
#SBATCH --mail-user=gianluca.mittone@unito.it
#SBATCH --mail-type=ALL
#SBATCH --time=00:30:00
#SBATCH --account=IscrC_NCCL-G
#SBATCH --partition=boost_usr_prod
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --exclusive

NAME=$1
PROCESSES_PER_NODE=$2 
FS=$3
MODEL=$4
DATASET=$5
SEED=$6 

source /leonardo_scratch/fast/uToID_bench/xffl/.venv/bin/activate

xffl --debug \
	simulate training.py \
		--processes-per-node "${PROCESSES_PER_NODE}" \
		--nodelist $(scontrol show hostnames "${SLURM_JOB_NODELIST}" | tr -s \\n " ") \
		--arguments --debug \
			--model "${MODEL}" \
			--dataset "${DATASET}" \
			--seed "${SEED}" \
			-hsdp 1 \
			-sub 16384 \
			-wb \
			-name "${NAME}" \
			-mode offline