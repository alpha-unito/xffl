#!/bin/bash

set -euo pipefail

# input environment variables:
#  - XFFL_MODEL_FOLDER
#  - XFFL_DATASET_FOLDER
#  - XFFL_IMAGE
#  - XFFL_FACILITY
#  - XFFL_OUTPUT_FOLDER
#  - XFFL_TMPDIR_FOLDER

ENVIRONMENT=""
PREFIX=${PREFIX:-""}
XFFL_LOCAL_TMPDIR=${TMPDIR:-""}
XFFL_OUTPUT_FOLDER=${XFFL_OUTPUT_FOLDER:-$XFFL_LOCAL_TMPDIR}
XFFL_EXECUTION=${XFFL_EXECUTION:-"false"}
XFFL_VENV=${XFFL_VENV:-"false"}
XFFL_IMAGE=${XFFL_IMAGE:-""}
XFFL_MODEL_FOLDER=${XFFL_MODEL_FOLDER:-""}
XFFL_DATASET_FOLDER=${XFFL_DATASET_FOLDER:-""}

if [[ "${XFFL_FACILITY}" == dummy* ]]; then
  cp -r ${XFFL_MODEL_FOLDER} output # FIXME: hardcoded
  EXIT_CODE=$?
  exit $EXIT_CODE
fi

XFFL_SCRIPTS_FOLDER="$(dirname "$0")"
source "${XFFL_SCRIPTS_FOLDER}/env.sh"

# Set specific facility env variables if xffl exec
if [ "${XFFL_EXECUTION}" = "true" ] ; then
	if [ "${XFFL_FACILITY}" != "None" ]; then
		if [ ! -f "${XFFL_FACILITY}" ]; then
			echo "Specified facility does not exist (${XFFL_FACILITY}) - the environment will not be initialized."
		else
			source "$XFFL_FACILITY"
		fi
	fi
fi

# Set general env variables for distributed ML
if [ -n "${XFFL_IMAGE}" ] ; then
	Container_platform_detection
	export XFFL_VENV=""
else
	export CONTAINER_PLT=""
fi

Derive_env
Limit_PyTorch_threads
Reset_visible_devices
LLaMA_default_env
Gpu_detection

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

		XFFL_TASKSET="taskset --cpu-list "$(( LOCAL_RANK * OMP_NUM_THREADS ))"-"$(( LOCAL_RANK * OMP_NUM_THREADS + OMP_NUM_THREADS - 1))

		if [ -n "${XFFL_VENV}" ] ; then # Python virtual environment
			XFFL_RANKS="RANK=$RANK LOCAL_RANK=$LOCAL_RANK ROLE_RANK=$ROLE_RANK GROUP_RANK=$GROUP_RANK"

			COMMAND="${XFFL_RANKS} ${XFFL_TASKSET} python $*"
		else # Container image
			export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}XFFL_IMAGE=${XFFL_IMAGE}"

			XFFL_RANKS="${PREFIX}RANK=${RANK} ${PREFIX}LOCAL_RANK=${LOCAL_RANK} ${PREFIX}ROLE_RANK=${ROLE_RANK} ${PREFIX}GROUP_RANK=${GROUP_RANK}"

			if [ -n "${XFFL_MODEL_FOLDER}" ] ; then
				MOUNT_XFFL_MODEL_FOLDER="--mount type=bind,src=${XFFL_MODEL_FOLDER}/,dst=/model/"
			else
				MOUNT_XFFL_MODEL_FOLDER=""
			fi

			if [ ! -n "${XFFL_DATASET_FOLDER}" ] ; then
				MOUNT_XFFL_DATASET_FOLDER="--mount type=bind,src=${XFFL_DATASET_FOLDER}/,dst=/dataset/"
			else
				MOUNT_XFFL_DATASET_FOLDER=""
			fi

			COMMAND="
${ENVIRONMENT} \
${XFFL_RANKS} \
${XFFL_TASKSET} \
${CONTAINER_PLT} exec \
${MOUNT_XFFL_MODEL_FOLDER} \
${MOUNT_XFFL_DATASET_FOLDER} \
--mount type=bind,src=${XFFL_LOCAL_TMPDIR}/,dst=/tmp/ \
--mount type=bind,src=${XFFL_OUTPUT_FOLDER}/,dst=/output/ \
--mount type=bind,src=${XFFL_CODE_FOLDER}/,dst=/code/ \
--cleanenv \
--home /code/ \
${GPU_FLAG} \
${XFFL_IMAGE} \
bash -c \"python /code/$* --model /model/ --dataset /dataset/\""
		fi

		# Run the local simulation process
		eval "$COMMAND" &
		pids[_RANK]=$!
	done

	# Wait for all processes to terminate
	for pid in "${pids[@]}" ; do
    	wait "$pid"
	done

# StreamFlow execution
else
	export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}XFFL_IMAGE=${XFFL_IMAGE}"
	export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"

	XFFL_RESOLVED_MODEL_FOLDER=$(readlink -f "${XFFL_MODEL_FOLDER}")
	XFFL_RESOLVED_DATASET_FOLDER=$(readlink -f "${XFFL_DATASET_FOLDER}")
	XFFL_RESOLVED_CODE_FOLDER=$(readlink -f "$1")
	XFFL_RESOLVED_CODE_FOLDER=$(dirname "${XFFL_RESOLVED_CODE_FOLDER}")
	# TODO: mount the whole workdir. It avoids to mount each single path to include the data are created by the step
	#	however, if the data are not in the workdir (e.g. the dataset). It is necessary to mount also the real path
	COMMAND="
${ENVIRONMENT} \
${CONTAINER_PLT} exec \
--mount type=bind,src=${XFFL_RESOLVED_MODEL_FOLDER}/,dst=/model/ \
--mount type=bind,src=${XFFL_RESOLVED_DATASET_FOLDER}/,dst=/dataset/ \
--mount type=bind,src=${XFFL_LOCAL_TMPDIR}/,dst=/tmp/ \
--mount type=bind,src=${XFFL_OUTPUT_FOLDER}/,dst=/output/ \
--mount type=bind,src=${XFFL_RESOLVED_CODE_FOLDER}/,dst=/code \
--cleanenv \
--home /home/ \
${GPU_FLAG} \
${XFFL_IMAGE} \
bash -c \"python /code/$(basename $1)\""

	echo "[Rank $RANK] Executing: $COMMAND"
	eval "$COMMAND"
fi
