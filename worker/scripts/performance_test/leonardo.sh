#!/bin/bash

WORKDIR=/leonardo_scratch/fast/uToID_bench/23_llama_sc24
WANDB=leonardo
HPC=leonardo

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

	for NODES in 1 2 4 8 16 32 64 128 256;
	do
		RUN_NAME=${MODEL}-${NODES}x$(( NODES * 4))
		TIME=00:10:00
		
		QOS=
		if (( $NODES > 64 )); then
			QOS="--qos=boost_qos_bprod"
		fi
		
		echo "sbatch $QOS --nodes $NODES --ntasks $(( NODES * 4)) --time=$TIME --output ${LOGS_DIR}/${MODEL}/${RUN_NAME}.out --error ${LOGS_DIR}/${MODEL}/${RUN_NAME}.err --job-name $RUN_NAME ${SCRIPTS_DIR}/${HPC}.slurm -m /llama/worker/workspace/${MODEL} -t 1024 -v 1024 -w $WANDB -s 42 -p /llama/worker/workspace/profiler/${MODEL}/${RUN_NAME}"

		sbatch $QOS --nodes $NODES --ntasks $(( NODES * 4)) --time=$TIME --output ${LOGS_DIR}/${MODEL}/${RUN_NAME}.out --error ${LOGS_DIR}/${MODEL}/${RUN_NAME}.err --job-name $RUN_NAME ${SCRIPTS_DIR}/${HPC}.slurm -m /llama/worker/workspace/${MODEL} -t 1024 -v 1024 -w $WANDB -s 42 -p /llama/worker/workspace/profiler/${MODEL}/${RUN_NAME}
	done
done
