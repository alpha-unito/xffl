#!/bin/bash

# PRELOADING
if [ -n "$VENV" ] ; then
	find "$(dirname "$(dirname "${VENV}")")" -type f -exec cat {} + > /dev/null &
else
	find "${XFFL_IMAGE}" -type f -exec cat {} + > /dev/null &
fi

# input environment variables: 
#  - CODE_FOLDER 
#  - MODEL_FOLDER
#  - DATASET_FOLDER
#  - IMAGE
#  - FACILITY
#  - OUTPUT_FOLDER
#  - EXECUTABLE_FOLDER

XFFL_SCRIPTS_FOLDER="$(dirname "$0")"
source "${XFFL_SCRIPTS_FOLDER}/env.sh"

# Set general env variables for distributed ML
Derive_env
Limit_PyTorch_threads
Reset_visible_devices
LLaMA_default_env
Gpu_detection
Container_platform_detection

# Set specific facility env variables
XFFL_FACILITY_SCRIPT="${XFFL_SCRIPTS_FOLDER}/facilities/${XFFL_FACILITY}.sh"
if [ ! -f "${XFFL_FACILITY_SCRIPT}" ]; then
  echo "${XFFL_FACILITY_SCRIPT} does not exist."
  exit 1
fi
source "${XFFL_FACILITY_SCRIPT}"

if [ -z "${XFFL_OUTPUT_FOLDER}" ] ; then 
	XFFL_OUTPUT_FOLDER=$XFFL_LOCAL_TMPDIR
fi

# Local simulation
if [ "${XFFL_FACILITY}" = "local" ] ; then
	pids=()
	for _RANK in $( seq 0 1 $(( WORLD_SIZE - 1 )) ) ; do
		XFFL_RANKS="RANK=\"${_RANK}\" \
			LOCAL_RANK=\"${_RANK}\" \
			ROLE_RANK=\"${_RANK}\" \
			GROUP_RANK=\"${_RANK}\""
		XFFL_TASKSET="taskset --cpu-list "$(( _RANK * OMP_NUM_THREADS ))"-"$(( _RANK * OMP_NUM_THREADS + OMP_NUM_THREADS - 1))
		XFFL_RUN="xffl/workflow/scripts/run.sh"

		# Python virtual environment
		if [ -n "$VENV" ] ; then
			source "${VENV}"
			COMMAND="$(pip show xffl | grep Location | awk '{print $2}')/${XFFL_RUN}"
		else
		# Container image
			COMMAND="${CONTAINER_PLT} exec \
				--mount type=bind,src=${XFFL_CODE_FOLDER}/,dst=/code/ \
				--mount type=bind,src=${XFFL_MODEL_FOLDER}/,dst=/model/ \
				--mount type=bind,src=${XFFL_DATASET_FOLDER},dst=/datasets/ \
				--mount type=bind,src=${XFFL_LOCAL_TMPDIR}/,dst=/tmp/ \
				--mount type=bind,src=${XFFL_OUTPUT_FOLDER}/,dst=/output/ \
				--home /code/ \
				$GPU_FLAG \
				${XFFL_IMAGE} \
				/code/${XFFL_RUN}"
		fi

		# Run the local simulation process
		eval "${XFFL_RANKS} ${XFFL_TASKSET} $COMMAND $*" &
		pids[_RANK]=$!
	done

	# Wait for all processes to terminate
	for pid in "${pids[@]}" ; do
    	wait "$pid"
	done

# StreamFlow execution
else
	COMMAND="${CONTAINER_PLT} exec \
		--mount type=bind,src=${XFFL_MODEL_FOLDER},dst=/model \
		--mount type=bind,src=${XFFL_DATASET_FOLDER},dst=/datasets \
		--mount type=bind,src=${XFFL_LOCAL_TMPDIR},dst=/tmp \
		--mount type=bind,src=${XFFL_OUTPUT_FOLDER},dst=${XFFL_OUTPUT_FOLDER} \
		--mount type=bind,src=${XFFL_EXECUTABLE_FOLDER},dst=${XFFL_EXECUTABLE_FOLDER} \
		--mount type=bind,src=${XFFL_SCRIPTS_FOLDER},dst=${XFFL_SCRIPTS_FOLDER} \
		--mount type=bind,src=/leonardo/home/userexternal/amulone1/xffl/,dst=/leonardo/home/userexternal/amulone1/xffl/ \
		--home /tmp/ \
		$GPU_FLAG \
		${XFFL_IMAGE} \
		${XFFL_SCRIPTS_FOLDER}/run.sh $*"
	echo "[Rank $RANK] Executing: $COMMAND"		
	eval "$COMMAND"
fi
