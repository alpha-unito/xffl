#!/bin/bash

source "$(dirname "$0")/parser.sh"

############################################################
# Main program											 #
############################################################
echo "[Rank $RANK] Evaluating CLI parameters and pre-loading model and datasets..."
EXECUTABLE=$1; shift
Parser "$@"
cat "/model/"* > /dev/null & # Caching for improved performance
cat "/datasets/"* > /dev/null & # Caching for improved performance

COMMAND="time python /code/${EXECUTABLE} --model /model/ --dataset /datasets/"
echo "[Rank $RANK] $COMMAND"		
eval "$COMMAND"
