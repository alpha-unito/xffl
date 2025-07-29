#!/bin/bash

############################################################
# Main program											   #
############################################################

EXECUTABLE=$1; shift

if [ "${XFFL_SIMULATION}" = "true" ] ; then
	if [ -n "${XFFL_VENV}" ] ; then
	  if [[ $RANK -eq 0 ]]; then
      #COMMAND="nsys profile --trace=osrt,mpi,cuda --cuda-memory-usage=true --nic-metrics=true python ${EXECUTABLE} $*"
      COMMAND="python ${EXECUTABLE} $*"
    else
      COMMAND="python ${EXECUTABLE} $*"
    fi
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