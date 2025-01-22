#!/bin/bash

source "$(dirname "$0")/parser.sh"

############################################################
# Main program											 #
############################################################
echo "[Rank $RANK] Evaluating CLI parameters and pre-loading model and datasets..."
Parser "$@"
cat "/model/"* > /dev/null & # Caching for improved performance
cat "/datasets/"* > /dev/null & # Caching for improved performance

export PYTHONPATH=/code/examples/llama/worker/libraries/llama-recipes/src/:$PYTHONPATH # TODO: remove
COMMAND="time python /code/examples/llama/client/src/training.py --model /model/ --dataset /datasets/"
echo "[Rank $RANK] $COMMAND"		
eval "$COMMAND"
