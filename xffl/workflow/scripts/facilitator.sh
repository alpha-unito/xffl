#!/bin/bash -ex

# input environment variables: 
#  - XFFL_MODEL_FOLDER
#  - XFFL_DATASET_FOLDER
#  - XFFL_IMAGE
#  - XFFL_FACILITY
#  - XFFL_OUTPUT_FOLDER
#  - XFFL_TMPDIR_FOLDER
# TODO: update this list

XFFL_SCRIPTS_FOLDER="$(dirname "$0")"
source "${XFFL_SCRIPTS_FOLDER}/env.sh"

# Set specific facility env variables
XFFL_FACILITY_SCRIPT="${XFFL_SCRIPTS_FOLDER}/facilities/${XFFL_FACILITY}.sh"
if [ ! -f "${XFFL_FACILITY_SCRIPT}" ]; then
    echo "${XFFL_FACILITY_SCRIPT} does not exist."
	exit 1
else
    source "$XFFL_FACILITY_SCRIPT"
fi

# Set general env variables for distributed ML
Derive_env
Limit_PyTorch_threads
Reset_visible_devices
LLaMA_default_env
Gpu_detection
if [ -z "$XFFL_VENV" ] ; then
	Container_platform_detection
fi

if [ -z "${XFFL_OUTPUT_FOLDER}" ] ; then 
	XFFL_OUTPUT_FOLDER=$XFFL_LOCAL_TMPDIR
fi

# Local simulation
if [ "${XFFL_EXECUTION}" = "true" ] ; then
	pids=()

	if [ -n "${XFFL_VENV}" ] ; then
		source "${XFFL_VENV}/bin/activate"
	fi

	for _RANK in $( seq $(( XFFL_NODEID * LOCAL_WORLD_SIZE )) 1 $(( XFFL_NODEID * LOCAL_WORLD_SIZE + LOCAL_WORLD_SIZE - 1 )) ) ; do
		RANK=$_RANK
		LOCAL_RANK=$(( _RANK % LOCAL_WORLD_SIZE ))
		ROLE_RANK=$_RANK
		GROUP_RANK=$(( _RANK / LOCAL_WORLD_SIZE ))

		XFFL_RANKS="RANK=$RANK LOCAL_RANK=$LOCAL_RANK ROLE_RANK=$ROLE_RANK GROUP_RANK=$GROUP_RANK"
		XFFL_TASKSET="taskset --cpu-list "$(( LOCAL_RANK * OMP_NUM_THREADS ))"-"$(( LOCAL_RANK * OMP_NUM_THREADS + OMP_NUM_THREADS - 1))

		# Python virtual environment
		if [ -n "$XFFL_VENV" ] ; then
			COMMAND="${XFFL_SCRIPTS_FOLDER}/run.sh $*"
		else
		# Container image
			COMMAND="${CONTAINER_PLT} exec \
--mount type=bind,src=${XFFL_MODEL_FOLDER}/,dst=/model/ \
--mount type=bind,src=${XFFL_DATASET_FOLDER},dst=/dataset/ \
--mount type=bind,src=${XFFL_LOCAL_TMPDIR}/,dst=/tmp/ \
--mount type=bind,src=${XFFL_OUTPUT_FOLDER}/,dst=/output/ \
--mount type=bind,src=${XFFL_CODE_FOLDER}/,dst=/code/ \
--home /code/ \
$GPU_FLAG \
${XFFL_IMAGE} \
bash -c \"python /code/$* --model /model/ --dataset /dataset/\""
		fi

		# Run the local simulation process
		eval "${XFFL_RANKS} ${XFFL_TASKSET} $COMMAND" &
		pids[_RANK]=$!
	done

	# Wait for all processes to terminate
	for pid in "${pids[@]}" ; do
    	wait "$pid"
	done

# StreamFlow execution
else
	XFFL_RESOLVED_MODEL_FOLDER=$(readlink -f "${XFFL_MODEL_FOLDER}")
	XFFL_RESOLVED_DATASET_FOLDER=$(readlink -f "${XFFL_DATASET_FOLDER}")
	XFFL_RESOLVED_CODE_FOLDER=$(readlink -f "$1")
	XFFL_RESOLVED_CODE_FOLDER=$(dirname "${XFFL_RESOLVED_CODE_FOLDER}")
	# TODO: mount the whole workdir. It avoids to mount each single path to include the data are created by the step
	#	however, if the data are not in the workdir (e.g. the dataset). It is necessary to mount also the real path
	COMMAND="${CONTAINER_PLT} exec \
		--mount type=bind,src=${XFFL_RESOLVED_MODEL_FOLDER}/,dst=/model/ \
		--mount type=bind,src=${XFFL_RESOLVED_DATASET_FOLDER}/,dst=/dataset/ \
		--mount type=bind,src=${XFFL_LOCAL_TMPDIR}/,dst=/tmp/ \
		--mount type=bind,src=${XFFL_OUTPUT_FOLDER}/,dst=/output/ \
		--mount type=bind,src=${XFFL_RESOLVED_CODE_FOLDER}/,dst=/code \
		--home /home/ \
		${GPU_FLAG} \
		${XFFL_IMAGE} \
		bash -c \"python /code/$(basename $1)\""
	echo "[Rank $RANK] Executing: $COMMAND"		# TODO: potremmo anche sbarazzarci di run.sh
	eval "$COMMAND"
fi