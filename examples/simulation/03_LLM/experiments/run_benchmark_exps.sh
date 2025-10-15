#!/bin/bash

WORKDIR=/leonardo_scratch/fast/uToID_bench/xffl/examples/simulation/03_LLM

# This script runs the experiments for the LLM simulation

MODEL="llama3.1-8b"
DATASET="clean_mc4_it"
SEED=42
ITERATIONS=3

# Single-node simulations
NODES=1
FS=1

echo "--- Single node simulations ---"

QOS=boost_qos_dbg
for PROCESSES_PER_NODE in 2 4; do
	echo "Running simulation with $NODES nodes and $PROCESSES_PER_NODE processes per node and $FS shards"
	NAME="${MODEL}"_ns_"${NODES}"_fs_"${FS}"_ppn_"${PROCESSES_PER_NODE}"
	if [ ! -f "${PWD}/logs/${NAME}.csv"  ]; then
		echo "sbatch --nodes $NODES --qos $QOS --job-name "${NAME}" --output "${WORKDIR}"/logs/"${NAME}".out --error "${WORKDIR}"/logs/"${NAME}".err leonardo.slurm "$NAME" "$PROCESSES_PER_NODE" $FS $MODEL $DATASET $SEED $ITERATIONS"
		sbatch --nodes $NODES --qos $QOS --job-name "${NAME}" --output "${WORKDIR}"/logs/"${NAME}".out --error "${WORKDIR}"/logs/"${NAME}".err leonardo.slurm "$NAME" "$PROCESSES_PER_NODE" $FS $MODEL $DATASET $SEED $ITERATIONS
	else
		echo "Skipping experiment since ${NAME} log already exists."
	fi
	printf "\n"
done

# Single-GPU multiple-node simulations
PROCESSES_PER_NODE=1
FS=1

echo "--- Single-GPU multiple-node simulations ---"

QOS=normal
for NODES in 2 4 8 16 32 64 128; do # excluded 256 nodes due to instability and costs
	echo "Running simulation with $NODES nodes and $PROCESSES_PER_NODE processes per node and $FS shards"
	NAME="${MODEL}"_ns_"${NODES}"_fs_"${FS}"_ppn_"${PROCESSES_PER_NODE}"
	if [ "$NODES" -gt 64 ]; then
		QOS=boost_qos_bprod
	fi
	if [ ! -f "${PWD}/logs/${NAME}.csv"  ]; then
		echo "sbatch --nodes $NODES --qos $QOS --job-name "${NAME}" --output "${WORKDIR}"/logs/"${NAME}".out --error "${WORKDIR}"/logs/"${NAME}".err leonardo.slurm "$NAME" "$PROCESSES_PER_NODE" $FS $MODEL $DATASET $SEED $ITERATIONS"
		sbatch --nodes $NODES --qos $QOS --job-name "${NAME}" --output "${WORKDIR}"/logs/"${NAME}".out --error "${WORKDIR}"/logs/"${NAME}".err leonardo.slurm "$NAME" "$PROCESSES_PER_NODE" $FS $MODEL $DATASET $SEED $ITERATIONS
	else
		echo "Skipping experiment since ${NAME} log already exists."
	fi
	printf "\n"
done

# Multiple-GPU multiple-node simulations
MODEL="llama3.1-70b"

PROCESSES_PER_NODE=4
FS=4

echo "--- Multiple-GPU multiple-node simulations ---"

QOS=normal
for NODES in 2 4 8 16 32 64 128; do # excluded 256 nodes due to instability and costs
	echo "Running simulation with $NODES nodes and $PROCESSES_PER_NODE processes per node and $FS shards"
	NAME="${MODEL}"_ns_"${NODES}"_fs_"${FS}"_ppn_"${PROCESSES_PER_NODE}"
	if [ "$NODES" -gt 64 ]; then
		QOS=boost_qos_bprod
	fi
	if [ ! -f "${PWD}/logs/${NAME}.csv"  ]; then
		echo "sbatch --nodes $NODES --qos $QOS --job-name "${NAME}" --output "${WORKDIR}"/logs/"${NAME}".out --error "${WORKDIR}"/logs/"${NAME}".err leonardo.slurm "$NAME" "$PROCESSES_PER_NODE" $FS $MODEL $DATASET $SEED $ITERATIONS"
		sbatch --nodes $NODES --qos $QOS --job-name "${NAME}" --output "${WORKDIR}"/logs/"${NAME}".out --error "${WORKDIR}"/logs/"${NAME}".err leonardo.slurm "$NAME" "$PROCESSES_PER_NODE" $FS $MODEL $DATASET $SEED $ITERATIONS
	else
		echo "Skipping experiment since ${NAME} log already exists."
	fi
	printf "\n"
done
