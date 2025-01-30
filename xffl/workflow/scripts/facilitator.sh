#!/bin/bash

# input environment variables: 
#  - CODE_FOLDER 
#  - MODEL_FOLDER
#  - DATASET_FOLDER
#  - IMAGE
#  - FACILITY

source "$(dirname "$0")/env.sh"

# Set general env variables for distributed ML
Derive_env
Limit_PyTorch_threads
Reset_visible_devices
LLaMA_default_env
Gpu_detection
Container_platform_detection

# Set specific facility env variables
FACILITY_SCRIPT="$(dirname "$0")/facilities/${FACILITY}.sh"
if [ ! -f "${FACILITY_SCRIPT}" ]; then
  echo "${FACILITY_SCRIPT} does not exist."
  exit 1
fi
source "${FACILITY_SCRIPT}"

# Local simulation
if [ "${FACILITY}" = "local" ] ; then
	pids=()
	for _RANK in $( seq 0 1 $(( WORLD_SIZE - 1 )) ) ; do
		XFFL_RANKS="RANK=\"${_RANK}\" \
				LOCAL_RANK=\"${_RANK}\" \
				ROLE_RANK=\"${_RANK}\" \
				GROUP_RANK=\"${_RANK}\""
		XFFL_TASKSET="taskset --cpu-list "$(( _RANK * OMP_NUM_THREADS ))"-"$(( _RANK * OMP_NUM_THREADS + OMP_NUM_THREADS - 1))

		if [ -n "$VENV" ] ; then
			source "${VENV}"
			COMMAND="$(pip show xffl | grep Location | awk '{print $2}')/xffl/workflow/scripts/run.sh $*"
		else
			COMMAND="${CONTAINER_PLT} exec \
				--mount type=bind,src=${CODE_FOLDER}/,dst=/code/ \
				--mount type=bind,src=${MODEL_FOLDER}/,dst=/model/ \
				--mount type=bind,src=${DATASET_FOLDER},dst=/datasets/ \
				--mount type=bind,src=${LOCAL_TMPDIR}/,dst=/tmp/ \
				--home /code/ \
				$GPU_FLAG \
				$IMAGE \
				/code/xffl/workflow/scripts/run.sh $*"
		fi
		eval "$XFFL_RANKS $XFFL_TASKSET $COMMAND" &
		pids[_RANK]=$!
	done
	for pid in "${pids[@]}" ; do
    	wait "$pid"
	done

# StreamFlow execution
else
	COMMAND="${CONTAINER_PLT} exec \
		--mount type=bind,src=${CODE_FOLDER}/,dst=/code/ \
		--mount type=bind,src=${MODEL_FOLDER}/,dst=/model/ \
		--mount type=bind,src=${DATASET_FOLDER},dst=/datasets/ \
		--mount type=bind,src=${LOCAL_TMPDIR}/,dst=/tmp/ \
		--home /code/ \
		$GPU_FLAG \
		$IMAGE \
		/code/xffl/workflow/scripts/run.sh $*"
	echo "[Rank $RANK] Executing: $COMMAND"		
	eval "$COMMAND"
fi
