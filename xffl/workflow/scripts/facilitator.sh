#!/bin/bash

# PRELOADING
if [ -n "$VENV" ] ; then
	find "$(dirname "$(dirname "${VENV}")")" -type f -exec cat {} + > /dev/null &
else
	find "${XFFL_IMAGE}" -type f -exec cat {} + > /dev/null &
fi

# input environment variables: 
#  - XFFL_MODEL_FOLDER
#  - XFFL_DATASET_FOLDER
#  - XFFL_IMAGE
#  - XFFL_FACILITY
#  - XFFL_OUTPUT_FOLDER
#  - XFFL_TMPDIR_FOLDER

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
	XFFL_EXECUTABLE_FOLDER=$(dirname $1)
	XFFL_RESOLVED_MODEL_FOLDER=$(readlink -f ${XFFL_MODEL_FOLDER})
	XFFL_RESOLVED_DATASET_FOLDER=$(readlink -f ${XFFL_DATASET_FOLDER})

	# TODO: mount the whole workdir. It avoids to mount each single path to incluce the data are created by the step
	#	however, if the data are not in the workdir (e.g. the dataset). It is necessary to mount also the real path
	COMMAND="${CONTAINER_PLT} exec \
		--mount type=bind,src=${XFFL_MODEL_FOLDER},dst=${XFFL_MODEL_FOLDER} \
		--mount type=bind,src=${XFFL_RESOLVED_MODEL_FOLDER},dst=${XFFL_RESOLVED_MODEL_FOLDER} \
		--mount type=bind,src=${XFFL_DATASET_FOLDER},dst=${XFFL_DATASET_FOLDER} \
		--mount type=bind,src=${XFFL_RESOLVED_DATASET_FOLDER},dst=${XFFL_RESOLVED_DATASET_FOLDER} \
		--mount type=bind,src=${XFFL_LOCAL_TMPDIR},dst=/tmp \
		--mount type=bind,src=${XFFL_OUTPUT_FOLDER},dst=${XFFL_OUTPUT_FOLDER} \
		--mount type=bind,src=${XFFL_TMPDIR_FOLDER}/,dst=${XFFL_TMPDIR_FOLDER}/ \
		--mount type=bind,src=${XFFL_EXECUTABLE_FOLDER},dst=${XFFL_EXECUTABLE_FOLDER} \
		--mount type=bind,src=${XFFL_SCRIPTS_FOLDER},dst=${XFFL_SCRIPTS_FOLDER} \
		--mount type=bind,src=/leonardo/home/userexternal/amulone1/xffl/,dst=/leonardo/home/userexternal/amulone1/xffl/ \
		--mount type=bind,src=/leonardo/home/userexternal/amulone1/,dst=/home/ \
		--home /home/ \
		${GPU_FLAG} \
		${XFFL_IMAGE} \
		${XFFL_SCRIPTS_FOLDER}/run.sh $*"
	echo "[Rank $RANK] Executing: $COMMAND"		
	eval "$COMMAND"
fi
