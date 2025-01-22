#!/bin/bash

# input environment variables: 
#  - XFFL_FOLDER 
#  - MODEL_FOLDER
#  - DATASET_FOLDER
#  - IMAGE
#  - FACILITY

source "$(dirname "$0")/env.sh"

# Set general env variables for distributed ML
Derive_env_from_SLURM
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

# Singularity container launch
COMMAND="${CONTAINER_PLT} exec \
	--mount type=bind,src=${CODE_FOLDER}/,dst=/code/ \
	--mount type=bind,src=${MODEL_FOLDER}/,dst=/model/ \
	--mount type=bind,src=${DATASET_FOLDER},dst=/datasets/ \
	--mount type=bind,src=${LOCAL_TMPDIR}/,dst=/tmp/ \
	--home /code/ \
	$GPU_FLAG \
	$IMAGE \
	/code/xffl/scripts/run.sh $*"
echo "[Rank $RANK] $COMMAND"		
eval "$COMMAND"
