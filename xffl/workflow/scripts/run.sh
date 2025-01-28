#!/bin/bash

source "$(dirname "$0")/parser.sh"

############################################################
# Main program											   #
############################################################
if [ -n "${VIRTUAL_ENV}" ] ; then
	COMMAND="time python ${EXECUTABLE} $*"
else
	echo "[Rank $RANK] Evaluating CLI parameters and pre-loading model and datasets..."
	Parser "$@"
	cat "/model/"* > /dev/null & # Caching for improved performance
	cat "/datasets/"* > /dev/null & # Caching for improved performance
	COMMAND="time python ${EXECUTABLE} --model /model/ --dataset /datasets/"
fi

echo "[Rank $RANK] Executing: $COMMAND"		
$COMMAND
