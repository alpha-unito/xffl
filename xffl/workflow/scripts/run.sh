#!/bin/bash

############################################################
# Main program											   #
############################################################

EXECUTABLE=$1; shift

if [ "${XFFL_FACILITY}" = "local" ] ; then
	if [ -n "${VENV}" ] ; then
		COMMAND="python ${EXECUTABLE} $*"
		eval "$COMMAND"
	else
		COMMAND="time python ${EXECUTABLE} --model /model/ --dataset /datasets/"
		PYTHONPATH=${PYTHONPATH}:/leonardo/home/userexternal/gmittone/.local/bin eval "$COMMAND" # TODO: Remove Path modification
	fi
else
	COMMAND="time python ${EXECUTABLE} $*"
	echo "[Rank $RANK] $COMMAND"
	PYTHONPATH=${PYTHONPATH}:/leonardo/home/userexternal/amulone1/.local/bin eval "$COMMAND" # TODO: Remove Path modification
fi