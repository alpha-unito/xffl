#!/bin/bash

WORKDIR=/leonardo/home/userexternal/gmittone/xffl/examples/simulation/03_LLM

# This script runs the experiments for the LLM simulation
MODEL="llama3.1-8b"
DATASET="clean_mc4_it"
SEED=42

# Simulations
NODES=128
PROCESSES_PER_NODE=1
FS=1
FB=8

QOS=normal
if [ "$NODES" -gt 64 ]; then
	QOS=boost_qos_bprod
fi

PREFIX="TEST"

# ----- FSDP ----

echo "Running FSDP with $NODES nodes and $PROCESSES_PER_NODE processes per node and $FS shards"
NAME="${PREFIX}"_FSDP_"${MODEL}"_"${DATASET}"_ns_"${NODES}"_ppn_"${PROCESSES_PER_NODE}"

echo "sbatch --nodes $NODES --qos $QOS --job-name ${NAME} --output ${WORKDIR}/new_logs/${NAME}.out --error ${WORKDIR}/new_logs/${NAME}.err learning_fsdp.slurm $NAME $PROCESSES_PER_NODE $FS $MODEL $DATASET $SEED $FB"
sbatch --nodes $NODES --qos $QOS --job-name "${NAME}" --output "${WORKDIR}"/new_logs/"${NAME}".out --error "${WORKDIR}"/new_logs/"${NAME}".err learning_fsdp.slurm "$NAME" "$PROCESSES_PER_NODE" "$FS" "$MODEL" "$DATASET" "$SEED" "$FB"

# ----- HSDP ----

echo "Running HSDP with $NODES nodes and $PROCESSES_PER_NODE processes per node and $FS shards"
NAME="${PREFIX}"_HSDP_"${MODEL}"_"${DATASET}"_ns_"${NODES}"_ppn_"${PROCESSES_PER_NODE}"

echo "sbatch --nodes $NODES --qos $QOS --job-name ${NAME} --output ${WORKDIR}/new_logs/${NAME}.out --error ${WORKDIR}/new_logs/${NAME}.err learning_hsdp.slurm $NAME $PROCESSES_PER_NODE $FS $MODEL $DATASET $SEED $FB"
sbatch --nodes $NODES --qos $QOS --job-name "${NAME}" --output "${WORKDIR}"/new_logs/"${NAME}".out --error "${WORKDIR}"/new_logs/"${NAME}".err learning_hsdp.slurm "$NAME" "$PROCESSES_PER_NODE" "$FS" "$MODEL" "$DATASET" "$SEED" "$FB"

# ----- FL ----

echo "Running FL with $NODES nodes and $PROCESSES_PER_NODE processes per node and $FS shards"
NAME="${PREFIX}"_FL_"${MODEL}"_"${DATASET}"_ns_"${NODES}"_ppn_"${PROCESSES_PER_NODE}"_fs_"${FS}"_fb_"${FB}"

echo "sbatch --nodes $NODES --qos $QOS --job-name ${NAME} --output ${WORKDIR}/new_logs/${NAME}.out --error ${WORKDIR}/new_logs/${NAME}.err learning_fl.slurm $NAME $PROCESSES_PER_NODE $FS $MODEL $DATASET $SEED $FB"
sbatch --nodes $NODES --qos $QOS --job-name "${NAME}" --output "${WORKDIR}"/new_logs/"${NAME}".out --error "${WORKDIR}"/new_logs/"${NAME}".err learning_fl.slurm "$NAME" "$PROCESSES_PER_NODE" "$FS" "$MODEL" "$DATASET" "$SEED" "$FB"