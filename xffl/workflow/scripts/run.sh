#!/bin/bash

source "$(dirname "$0")/parser.sh"

############################################################
# Main program											   #
############################################################

echo "[Rank $RANK] Evaluating CLI parameters and pre-loading model and datasets..."
EXECUTABLE=$1; shift
#Parser "$@"

if [ -n "${VENV}" ] ; then
	find "$MODEL_FOLDER" -type f -exec cat {} + > /dev/null & # Caching for improved performance
	find "$DATASET_FOLDER" -type f -exec cat {} + > /dev/null & # Caching for improved performance
	COMMAND="time python ${CODE_FOLDER}/${EXECUTABLE} $*"
	eval "$COMMAND"
else
	find "/model/" -type f -exec cat {} + > /dev/null & # Caching for improved performance
	find "/datasets/" -type f -exec cat {} + > /dev/null & # Caching for improved performance
	COMMAND="time python /code/${EXECUTABLE} --model /model/ --dataset /datasets/"
	PYTHONPATH=${PYTHONPATH}:/leonardo/home/userexternal/gmittone/.local/bin eval "$COMMAND"
fi


