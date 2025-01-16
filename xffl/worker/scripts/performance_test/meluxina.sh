#!/bin/bash

WORKDIR=/project/home/p200594
WANDB=meluxina
HPC=meluxina

###

LOGS_DIR=${WORKDIR}/worker/workspace/logs
PROFILER_DIR=${WORKDIR}/worker/workspace/profiler
SCRIPTS_DIR=${WORKDIR}/worker/scripts

mkdir ${LOGS_DIR}
mkdir ${PROFILER_DIR}

for MODEL in llama3.1-8b llama3.1-70b mixtral-8x7b-v0.1
do
	mkdir ${LOGS_DIR}/${MODEL}
	mkdir ${PROFILER_DIR}/${MODEL}

	for NODES in 1 2 4 8 16 32;
	do
		RUN_NAME=${MODEL}-${NODES}x$(( NODES * 4))
		TIME=01:00:00
		
		echo "sbatch --nodes $NODES --ntasks $(( NODES * 4)) --time=$TIME --output ${LOGS_DIR}/${MODEL}/${RUN_NAME}.out --error ${LOGS_DIR}/${MODEL}/${RUN_NAME}.err --job-name $RUN_NAME ${SCRIPTS_DIR}/${HPC}.slurm -m /llama/worker/workspace/${MODEL} -t 1024 -v 1024 -w $WANDB -s 42 -p /llama/worker/workspace/profiler/${MODEL}/${RUN_NAME}"

		sbatch --nodes $NODES --ntasks $(( NODES * 4)) --time=$TIME --output ${LOGS_DIR}/${MODEL}/${RUN_NAME}.out --error ${LOGS_DIR}/${MODEL}/${RUN_NAME}.err --job-name $RUN_NAME ${SCRIPTS_DIR}/${HPC}.slurm -m /llama/worker/workspace/${MODEL} -t 1024 -v 1024 -w $WANDB -s 42 -p /llama/worker/workspace/profiler/${MODEL}/${RUN_NAME}
	done
done
