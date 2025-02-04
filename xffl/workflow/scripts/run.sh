#!/bin/bash

############################################################
# Main program											   #
############################################################

EXECUTABLE=$1; shift

if [ "${FACILITY}" = "local" ] ; then
	if [ -n "${VENV}" ] ; then
		COMMAND="python ${EXECUTABLE} $*"
		eval "$COMMAND"
	else
		find "/model/" -type f -exec cat {} + > /dev/null & # Caching for improved performance
		find "/datasets/" -type f -exec cat {} + > /dev/null & # Caching for improved performance

		COMMAND="time python /code/${EXECUTABLE} --model /model/ --dataset /datasets/"
		PYTHONPATH=${PYTHONPATH}:/leonardo/home/userexternal/gmittone/.local/bin eval "$COMMAND" # TODO: Remove Path modification
	fi
else
	echo "[Rank $RANK] Pre-loading model and datasets..."
	find "/model/" -type f -exec cat {} + > /dev/null & # Caching for improved performance
	find "/datasets/" -type f -exec cat {} + > /dev/null & # Caching for improved performance

	COMMAND="time python /code/${EXECUTABLE} --model /model/ --dataset /datasets/ $*"
	echo "[Rank $RANK] $COMMAND"
	PYTHONPATH=${PYTHONPATH}:/leonardo/home/userexternal/amulone1/venv-xxfl/bin/xffl eval "$COMMAND" # TODO: Remove Path modification
fi