#!/bin/bash

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
if [ ! -n "$VENV" ] ; then
	Container_platform_detection
fi

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
if [ "${XFFL_SIMULATION}" = "true" ] ; then
	pids=()

	if [ -n "${VENV}" ] ; then
		source ${VENV}
	fi

	for _RANK in $( seq $(( XFFL_NODEID * LOCAL_WORLD_SIZE )) 1 $(( XFFL_NODEID * LOCAL_WORLD_SIZE + LOCAL_WORLD_SIZE - 1 )) ) ; do
		RANK=$_RANK
		LOCAL_RANK=$(( _RANK % LOCAL_WORLD_SIZE ))
		ROLE_RANK=$_RANK
		GROUP_RANK=$(( _RANK / LOCAL_WORLD_SIZE ))

		XFFL_RANKS="RANK=$RANK LOCAL_RANK=$LOCAL_RANK ROLE_RANK=$ROLE_RANK GROUP_RANK=$GROUP_RANK"
		XFFL_TASKSET="taskset --cpu-list "$(( LOCAL_RANK * OMP_NUM_THREADS ))"-"$(( LOCAL_RANK * OMP_NUM_THREADS + OMP_NUM_THREADS - 1))

		# Python virtual environment
		if [ -n "$VENV" ] ; then
			COMMAND="${XFFL_SCRIPTS_FOLDER}/run.sh"
		else
		# Container image
			COMMAND="${CONTAINER_PLT} exec \
				--mount type=bind,src=${XFFL_MODEL_FOLDER}/,dst=/model/ \
				--mount type=bind,src=${XFFL_DATASET_FOLDER},dst=/datasets/ \
				--mount type=bind,src=${XFFL_LOCAL_TMPDIR}/,dst=/tmp/ \
				--mount type=bind,src=${XFFL_OUTPUT_FOLDER}/,dst=/output/ \
				--mount type=bind,src=${XFFL_CODE_FOLDER}/,dst=/code/ \
				--home /code/ \
				$GPU_FLAG \
				${XFFL_IMAGE} \
				/usr/local/lib/python3.12/dist-packages/xffl/workflow/scripts/run.sh" # TODO: make this dynamic
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
	XFFL_RESOLVED_EXECUTABLE_FOLDER=$(readlink -f $1)
	XFFL_RESOLVED_EXECUTABLE_FOLDER=$(dirname ${XFFL_RESOLVED_EXECUTABLE_FOLDER})
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
		--mount type=bind,src=${XFFL_RESOLVED_EXECUTABLE_FOLDER},dst=${XFFL_RESOLVED_EXECUTABLE_FOLDER} \
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
