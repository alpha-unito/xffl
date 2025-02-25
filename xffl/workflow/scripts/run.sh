#!/bin/bash

############################################################
# Main program											   #
############################################################

EXECUTABLE=$1; shift

if [ "${XFFL_SIMULATION}" = "true" ] ; then
	if [ -n "${VENV}" ] ; then
		COMMAND="python ${EXECUTABLE} $*"
		eval "$COMMAND"
	else
		COMMAND="time python ${EXECUTABLE} --model /model/ --dataset /datasets/"
		eval "$COMMAND"
	fi
else
	COMMAND="time python ${EXECUTABLE} $*"
	echo "[Rank $RANK] $COMMAND"
	eval "$COMMAND"
fi