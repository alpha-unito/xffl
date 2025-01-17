#!/bin/bash

Help()
{
   # Display Help
   echo "LLaMA-3 runner script"
   echo
   echo "Syntax: exec_llama [-i|--image] [-f|--facility] [--help]"
   echo "options:"
   echo "-i|--image 		Singularity image path"
   echo "-f|--facility 		Facility [leonardo/lumi/meluxina/marenostrum]"
   echo "-m|--model 		Model path"
   echo "--tokenizer 		Tokenizer path"
   echo "-r|--repository	Path of repository which contains llama_repices"
   echo "-d|--dataset 		Dataset path"
   echo "--workdir 		    Working directory path"
   echo "--help    			Shows this message"
   echo
}

############################################################
# Process the input options. Add options as needed.        #
############################################################

OPTIONS=""
while [[ $# -gt 0 ]]; do
	case $1 in
		-f|--facility)
			FACILITY="$2"
			OPTIONS="$OPTIONS --facility $2"
			shift
			shift
			;;
		-m|--model)
			MODEL="$2"
			OPTIONS="$OPTIONS --model /models/$(basename $2)"
			shift
			shift
			;;
		--tokenizer)
			TOKENIZER="$2"
			OPTIONS="$OPTIONS --tokenizer /tokenizers/$(basename $2)"
			shift
			shift
			;;
		-r|--repository)
			REPOSITORY_DIR="$2"
			shift
			shift
			;;
		-w|--workdir)
			WORKDIR="$2"
			shift
			shift
			;;	
		--help)
			Help
			exit 0
			;;
		-*|--*)
			OPTIONS="$OPTIONS $1 $2"
			shift
			shift
			;;
		*)
			echo "exec_llama.sh: Unknown option $1. Use --help to get help."
			exit 1
			;;
	esac
done

if [[ -z "${IMAGE}" ]]
then
	echo "Option --image is not setted"
	exit 1
fi

if [[ -z "${FACILITY}" ]]
then
	echo "Option --facility is not setted"
	exit 1
fi

if [[ -z "${REPOSITORY_DIR}" ]]
then
	echo "Option --repository is not setted"
	exit 1
fi


echo "OPTIONS: $OPTIONS"

RESOLVED_PATH=$(echo $(readlink -f ${REPOSITORY_DIR}) | xargs dirname)
echo "Original REPOSITORY_DIR path: ${REPOSITORY_DIR}\nResolved path: ${RESOLVED_PATH}"


cp -r $MODEL .
exit 0



# The PyTorch's FSDP runtime requires the following environment variables to be correctly set to operate correctly.
# These environment variables are very similar to those used by other distributed systems (i.e., MPI)
# We extract this information from the SLURM runtime environment through srun, but an equivalent setting is obtained through mpirun (mpi equivalent variables are indicated in comments)
if (( $SLURM_NTASKS_PER_NODE >= $SLURM_NTASKS )); then		# SLURM_GPUS_ON_NODE, OMPI_COMM_WORLD_LOCAL_SIZE
	export LOCAL_WORLD_SIZE=$SLURM_NTASKS
else
	export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE
fi
export WORLD_SIZE=$SLURM_NTASKS			 					# OMPI_COMM_WORLD_SIZE
export GROUP_WORLD_SIZE=$SLURM_JOB_NUM_NODES				# OMPI_MCA_orte_num_nodes 
export ROLE_WORLD_SIZE=$SLURM_NTASKS 						# OMPI_COMM_WORLD_SIZE
export LOCAL_RANK=$SLURM_LOCALID 							# OMPI_COMM_WORLD_LOCAL_RANK
export RANK=$SLURM_PROCID 									# OMPI_COMM_WORLD_RANK
export ROLE_RANK=$RANK										# $OMPI_COMM_WORLD_RANK
export GROUP_RANK=$(( RANK / SLURM_NTASKS_PER_NODE ))
export ROLE_NAME="default"
export MASTER_ADDR=$SLURM_SRUN_COMM_HOST
export MASTER_PORT=29500

# Due to PyTorch's aggressive thread policy OMP_NUM_THREADS should be manually set to the number of actually available cores (by default PyTorch would spawn a thread for each processor's core)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# To agevolate PyTorch's FSDP hybrid sharding is fundamental that each process knows how many other processes are allocated on the same machine (i.e., the number of local GPUs), so that inter-node sharding and inter-node replication is handled correctly
export CUDA_VISIBLE_DEVICES=$( seq -s , 0 $(( $SLURM_GPUS_PER_NODE - 1 )) )

# Default LLaMA variables - it's better to have them here than inside the llama-recipes code
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_DISABLE_ADDR2LINE=1
export NCCL_CROSS_NIC=1

LOCAL_TMPDIR="--mount type=bind,src=/tmp/,dst=/tmp/"
CONTAINER="singularity"

if [ "$1" = "lumi" ]; then	# LUMI
	# For compatibility with AMD GPUs; theoretically also CUDA_VISIBLE_DEVICES should work
	export HIP_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
	export ROCR_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES		
	GPU_FLAG="--rocm"

	# MIOPEN needs some initialisation for the cache as the default location
	# does not work on LUMI as Lustre does not provide the necessary features.
	export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
	export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
	if [ $SLURM_LOCALID -eq 0 ] ; then
		rm -rf $MIOPEN_USER_DB_PATH
		mkdir -p $MIOPEN_USER_DB_PATH
	fi

	# Set interfaces to be used by RCCL.
	# This is needed as otherwise RCCL tries to use a network interface it has
	# no access to on LUMI.
	export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
	export NCCL_NET_GDR_LEVEL=3
else
	GPU_FLAG="--nv"
	if [ "$1" = "karolina" ]; then
		CONTAINER="apptainer"
		export APPTAINER_BINDPATH=$SINGULARITY_BINDPATH
		export APPTAINERENV_LD_PRELOAD=$SINGULARITYENV_LD_PRELOAD
	elif [ "$1" = "meluxina" ]; then
		LOCAL_TMPDIR="--mount type=bind,src=${LOCALSCRATCH}/,dst=/tmp/"
	else # Leonardo and MareNostrum
		LOCAL_TMPDIR="--mount type=bind,src=${TMPDIR}/,dst=/tmp/"
	fi
fi

# Singularity container launch
echo "${CONTAINER} exec \
--mount type=bind,src=${ROOT_FOLDER}/,dst=/llama/ \
--mount type=bind,src=${ROOT_FOLDER}/datasets/,dst=/datasets/ \
$LOCAL_TMPDIR \
--home /llama/worker/workspace/ \
$GPU_FLAG \
${ROOT_FOLDER}/worker/workspace/worker.sif \
/llama/worker/scripts/torchrun.sh -h $@"


${CONTAINER} exec \
	--mount type=bind,src=${REPOSITORY_DIR}/,dst=/llama/ \
	--mount type=bind,src=${REPOSITORY_DIR}/datasets/,dst=/datasets/ \
	$LOCAL_TMPDIR \
	--home /llama/worker/workspace/ \
	$GPU_FLAG \
	${REPOSITORY_DIR}/worker/workspace/worker.sif \
	/llama/worker/scripts/torchrun.sh -h $@
