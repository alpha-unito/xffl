#!/bin/bash

WORKDIR=/leonardo_scratch/fast/IscrC_HOPE_0/23_llama_sc24
QOS=

cd ${WORKDIR}/worker/workspace
mkdir scalability_logs


for NODES in 1; #2 4 8 16 32 64; #128 256;
do
	if (( $NODES > 64 )); then
		QOS="--qos=boost_qos_bprod"
	fi

	if (( $NODES == 1 )); then
		TIME=08:00:00
	elif (( $NODES == 2 )); then
		TIME=08:00:00
	elif (( $NODES == 4 )); then
		TIME=06:00:00
	elif (( $NODES == 8 )); then
		TIME=04:00:00
	elif (( $NODES == 16 )); then
		TIME=03:00:00
	elif (( $NODES == 32 )); then
		TIME=02:00:00
	elif (( $NODES == 64 )); then
		TIME=01:30:00
	elif (( $NODES == 128 )); then
		TIME=01:00:00
	elif (( $NODES == 256 )); then
		TIME=01:00:00
	fi
	

	for REPLICA in 1; #2 3 4 5;
	do
		#sbatch $QOS --nodes $NODES --ntasks $(( NODES * 4)) --time=$TIME --output ${WORKDIR}/worker/workspace/scalability_logs/llama-${NODES}x$(( NODES * 4))_${REPLICA}.out --job-name llama-${NODES}x$(( NODES * 4))_${REPLICA} ${WORKDIR}/worker/scripts/leonardo.slurm -m llama-3-8b -t 100000 -v 5120 -w leonardo_${NODES}
		sbatch $QOS --nodes $NODES --ntasks $(( NODES * 4)) --time=$TIME --output ${WORKDIR}/worker/workspace/scalability_logs/llama-${NODES}x$(( NODES * 4))_${REPLICA}_x5.out --job-name llama-${NODES}x$(( NODES * 4))_${REPLICA}_x5 ${WORKDIR}/worker/scripts/leonardo.slurm -m llama-3-8b -t 100000 -v 5120 -w leonardo_${NODES}_x5 -s $RANDOM
		#sbatch $QOS --nodes $NODES --ntasks $(( NODES * 4)) --time=$TIME --output ${WORKDIR}/worker/workspace/scalability_logs/llama-${NODES}x$(( NODES * 4))_${REPLICA}_x10.out --job-name llama-${NODES}x$(( NODES * 4))_${REPLICA}_x10 ${WORKDIR}/worker/scripts/leonardo.slurm -m llama-3-8b -t 200000 -v 10240 -w leonardo_${NODES}_x10
	done
done
