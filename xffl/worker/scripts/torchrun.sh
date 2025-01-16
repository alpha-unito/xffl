#!/bin/bash

############################################################
# Help													 #
############################################################
Help()
{
	# Display Help
	echo "LLaMA-3 runner script"
	echo
	echo "Syntax: exec_llama [-h|--hpc] [-m|--model] [-t|--train] [-v|--validation] [-w|--wandb] [-e|--epochs] [-o|--output] [--help]"
	echo "options:"
	echo "-h|--hpc			HPC facility [leonardo/lumi/meluxina]"
	echo "-m|--model		LLaMA-3 model version [8b/70b]"
	echo "-t|--train		Number of training samples to load"
	echo "-v|--validation	Number of validation samples to load"
	echo "-w|--wandb		WandB group name of the run"
	echo "-e|--epochs		Training epochs"
	echo "-o|--output		Path to save the model"
	echo "-t|--tokenizer	Path of tokenizer"
	echo "-h|--help			Shows this message"
	echo
}

#########################################################
# Process the input options. Add options as needed.		#
#########################################################

while [[ $# -gt 0 ]]; do
	case $1 in
		-h|--hpc)
			HPC="$2"
			shift
			shift
			;;
		-m|--model)
			MODEL="--model_name $2"
			MODEL_NAME="$2"
			cat $2/* > /dev/null &
			shift
			shift
			;;
		-t|--train)
			TRAIN_SAMPLES="--train_samples $2"
			shift
			shift
			;;
		-v|--validation)
			VAL_SAMPLES="--test_samples $2"
			shift
			shift
			;;
		-w|--wandb)
			WANDB="--use_wandb --group $2"
			shift
			shift
			;;
		-e|--epochs)
			EPOCHS="--num_epochs $2"
			shift
			shift
			;;
		-o|--output)
			OUTPUT="--save_model --output_dir $2 --dist_checkpoint_root_folder $2"
			shift
			shift
			;;
		-t|--tokenizer)
			TOKENIZER="--tokenizer_name $2"
			shift
			shift
			;;
		-r|--replica)
			REPLICA="--hsdp --sharding_group_size $2 --replica_group_size $(( WORLD_SIZE / $2 ))"
			shift
			shift
			;;
		-s|--seed)
			SEED="--seed $2"
			shift
			shift
			;;
		-p|--profiler-dir)
			PROFILER="--use_profiler --profiler_dir $2"
			shift
			shift
			;;
		--help)
			Help
			exit 0
			;;
		*|-*|--*)
			echo "Unknown option $1. Use --help to get help."
			exit 1
			;;
	esac
done

############################################################
# Main program											 #
############################################################
#echo "[Rank $RANK] Pre-loading the datasets..."
cat /datasets/clean_mc4_it_val.hf/* > /dev/null &
cat /datasets/clean_mc4_it_train.hf/* > /dev/null &
cat $2 > /dev/null &

export PYTHONPATH=/llama/worker/libraries/llama-recipes/src/:$PYTHONPATH
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

echo "[Rank $RANK] time python /llama/worker/libraries/llama-recipes/src/llama_recipes/finetuning.py --enable_fsdp --dataset clean_mc4_it_dataset $SEED $MODEL $EPOCHS $TRAIN_SAMPLES $VAL_SAMPLES $WANDB $OUTPUT $REPLICA $TOKENIZER $PROFILER"
		
time python /llama/worker/libraries/llama-recipes/src/llama_recipes/finetuning.py --enable_fsdp --dataset clean_mc4_it_dataset $SEED $MODEL $EPOCHS $TRAIN_SAMPLES $VAL_SAMPLES $WANDB $OUTPUT $REPLICA $TOKENIZER $PROFILER