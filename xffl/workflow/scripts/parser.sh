#!/bin/bash

############################################################
# Help													 #
############################################################
Help()
{
	# Display Help
	echo "xFFL runner script"
	echo
	echo "Syntax: exec_llama [-t|--train] [-v|--validation] [-e|--epochs] [-w|--wandb] [-o|--output] [-s|--seed] [-h|--help]"
	echo "options:"
	echo "-t|--train		Number of training samples to load"
	echo "-v|--validation	Number of validation samples to load"
	echo "-e|--epochs		Training epochs"
	echo "-w|--wandb		WandB group name of the run"
	echo "-o|--output		Path to save the model"
	echo "-s|--seed			Random execution seed"
	echo "-h|--help			show this help message and exit"
	echo
}

#########################################################
# Process the input options. Add options as needed.		#
#########################################################

Parser()
{
	while [[ $# -gt 0 ]]; do
		case $1 in
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
			-e|--epochs)
				EPOCHS="--num_epochs $2"
				shift
				shift
				;;
			-w|--wandb)
				WANDB="--use_wandb --group $2"
				shift
				shift
				;;
			-o|--output)
				OUTPUT="--save_model --output_dir $2 --dist_checkpoint_root_folder $2"
				shift
				shift
				;;
			-s|--seed)
				SEED="--seed $2"
				shift
				shift
				;;
			-h|--help)
				Help
				exit 0
				;;
			*)
				echo "Unknown option $1. Use --help to get help."
				exit 1
				;;
		esac
	done
}
