#!/bin/bash

source "$(dirname "$0")/parser.sh"

############################################################
# Main program											 #
############################################################
echo "[Rank $RANK] Evaluating CLI parameters and pre-loading model and datasets..."
Parser "$@"
cat "/model/"/* > /dev/null & # Caching for improved performance
cat "/dataset/"/* > /dev/null & # Caching for improved performance

export PYTHONPATH=/code/llama/worker/libraries/llama-recipes/src/:$PYTHONPATH # TODO: remove
COMMAND="time python /code/llama/worker/libraries/llama-recipes/src/llama_recipes/finetuning.py --enable_fsdp --model_name /model/ --dataset /dataset/ $TRAIN_SAMPLES $VAL_SAMPLES $EPOCHS $WANDB $OUTPUT $SEED"
echo "[Rank $RANK] $COMMAND"		
eval "$COMMAND"
