#!/bin/bash

WORKDIR=/leonardo_scratch/fast/uToID_bench/xffl/examples/simulation/03_LLM

# This script runs the experiments for the LLM simulation
MODEL="llama3.1-8b"
DATASET="clean_mc4_it"
SEED=42

# Simulations
NODES=$1

echo "--- FSDP multi-node simulations ---"

# Multiple-GPU multiple-node simulations
PROCESSES_PER_NODE=4
FS=1

echo "Running simulation with $NODES nodes and $PROCESSES_PER_NODE processes per node and $FS shards"
NAME="${MODEL}"_ns_"${NODES}"_fs_"${FS}"_ppn_"${PROCESSES_PER_NODE}"

QOS=normal
if [ "$NODES" -gt 64 ]; then
	QOS=boost_qos_bprod
fi

echo "sbatch --nodes $NODES --qos $QOS --job-name ${NAME} --output ${WORKDIR}/logs/${NAME}.out --error ${WORKDIR}/logs/${NAME}.err leonardo.slurm $NAME $PROCESSES_PER_NODE $FS $MODEL $DATASET $SEED"
sbatch --nodes $NODES --qos $QOS --job-name "${NAME}" --output "${WORKDIR}"/learning_logs/"${NAME}".out --error "${WORKDIR}"/learning_logs/"${NAME}".err learning.slurm "$NAME" "$PROCESSES_PER_NODE" $FS $MODEL $DATASET $SEED