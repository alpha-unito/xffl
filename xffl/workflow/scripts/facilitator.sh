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

EXECUTABLE=$1; shift

if [ "${FACILITY}" = "local" ] ; then
	pids=()
	for _RANK in $( seq 0 1 $(( WORLD_SIZE - 1 )) ) ; do
		if [ -n "${VENV}" ] ; then
			source "${VENV}"
			COMMAND="RANK=\"${_RANK}\" \
				LOCAL_RANK=\"${_RANK}\" \
				ROLE_RANK=\"${_RANK}\" \
				$(pip show xffl | grep Location | awk '{print $2}')/xffl/workflow/scripts/run.sh $EXECUTABLE $*"
		else
			COMMAND=" RANK=\"${_RANK}\" \
				LOCAL_RANK=\"${_RANK}\" \
				ROLE_RANK=\"${_RANK}\" \
				${CONTAINER_PLT} exec \
				--mount type=bind,src=${CODE_FOLDER}/,dst=/code/ \
				--mount type=bind,src=${MODEL_FOLDER}/,dst=/model/ \
				--mount type=bind,src=${DATASET_FOLDER},dst=/datasets/ \
				--mount type=bind,src=${LOCAL_TMPDIR}/,dst=/tmp/ \
				--home /code/ \
				$GPU_FLAG \
				$IMAGE \
				/code/xffl/workflow/scripts/run.sh /code/$EXECUTABLE $*"
		fi
		eval "$COMMAND" &
		pids[_RANK]=$!
	done
	for pid in "${pids[@]}" ; do
    	wait "$pid"
	done

	echo "TUTTI MORTI"

else
	# Singularity container launch
	COMMAND="${CONTAINER_PLT} exec \
		--mount type=bind,src=${CODE_FOLDER}/,dst=/code/ \
		--mount type=bind,src=${MODEL_FOLDER}/,dst=/model/ \
		--mount type=bind,src=${DATASET_FOLDER},dst=/datasets/ \
		--mount type=bind,src=${LOCAL_TMPDIR}/,dst=/tmp/ \
		--home /code/ \
		$GPU_FLAG \
		$IMAGE \
		/code/xffl/workflow/scripts/run.sh /code/$EXECUTABLE $*"
	echo "[Rank $RANK] Executing: $COMMAND"		
	eval "$COMMAND"
fi
