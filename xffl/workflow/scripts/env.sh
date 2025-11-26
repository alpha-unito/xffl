#!/bin/bash

# The PyTorch's FSDP runtime requires the following environment variables to be correctly set to operate correctly.
# These environment variables are very similar to those used by other distributed systems (i.e., MPI)
# We extract this information from the SLURM runtime environment through srun, but an equivalent setting is obtained through mpirun (mpi equivalent variables are indicated in comments)
Derive_env () {

    if [ "${XFFL_EXECUTION}" = "true" ] ; then
        export ROLE_NAME="default"
        export MASTER_PORT=29500
        export LOCAL_WORLD_SIZE=$(( XFFL_WORLD_SIZE / XFFL_NUM_NODES )) # We assume an equal allocation
        export WORLD_SIZE=$XFFL_WORLD_SIZE		 					
        export GROUP_WORLD_SIZE=$XFFL_NUM_NODES				
        export ROLE_WORLD_SIZE=$XFFL_WORLD_SIZE

        if [ -n "$CONTAINER_PLT" ] ; then
            export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}ROLE_NAME=${ROLE_NAME}"
            export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}MASTER_ADDR=${MASTER_ADDR}"
            export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}MASTER_PORT=${MASTER_PORT}"
            export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}"
            export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}WORLD_SIZE=${WORLD_SIZE}"
            export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}GROUP_WORLD_SIZE=${GROUP_WORLD_SIZE}"
            export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}ROLE_WORLD_SIZE=${ROLE_WORLD_SIZE}"
        fi					

    elif command -v srun > /dev/null ; then # Check SLURM
        export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}ROLE_NAME=default"
        export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}MASTER_PORT=29500"
        export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}MASTER_ADDR=${MASTER_ADDR}"

        if [ -z "${SLURM_NTASKS_PER_NODE+x}" ]; then 
            echo "SLURM_NTASKS_PER_NODE is unset"
            exit 1
        fi
        if [ -z "${SLURM_NTASKS+x}" ]; then 
            echo "SLURM_NTASKS is unset"
            exit 1
        fi
        if (( SLURM_NTASKS_PER_NODE >= SLURM_NTASKS )); then		# SLURM_GPUS_ON_NODE, OMPI_COMM_WORLD_LOCAL_SIZE
            export LOCAL_WORLD_SIZE=$SLURM_NTASKS
        else
            export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE
        fi
        export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}"
        export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}WORLD_SIZE=$SLURM_NTASKS"			 					# OMPI_COMM_WORLD_SIZE
        export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}GROUP_WORLD_SIZE=$SLURM_JOB_NUM_NODES"				# OMPI_MCA_orte_num_nodes 
        export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}ROLE_WORLD_SIZE=$SLURM_NTASKS" 						# OMPI_COMM_WORLD_SIZE

        export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}LOCAL_RANK=$SLURM_LOCALID" 							# OMPI_COMM_WORLD_LOCAL_RANK
        export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}RANK=$SLURM_PROCID" 									# OMPI_COMM_WORLD_RANK
        export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}ROLE_RANK=$RANK"										# $OMPI_COMM_WORLD_RANK
        GROUP_RANK=$(( RANK / SLURM_NTASKS_PER_NODE ))
        export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}GROUP_RANK=${GROUP_RANK}"
        
        export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}MASTER_ADDR=$SLURM_SRUN_COMM_HOST"
    fi

    return 0
}

# Due to PyTorch's aggressive thread policy OMP_NUM_THREADS should be manually set to the number of actually available cores (by default PyTorch would spawn a thread for each processor's core)
Limit_PyTorch_threads () {
    if [ "${XFFL_EXECUTION}" = "true" ] ; then
	    export OMP_NUM_THREADS=$(( $(nproc --all) / LOCAL_WORLD_SIZE ))
    elif command -v srun > /dev/null ; then # Check SLURM
        if [ -z "${SLURM_CPUS_PER_TASK+x}" ]; then
            echo "SLURM_CPUS_PER_TASK is unset"
            exit 1
        fi
        export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    else
        exit 1
    fi

    if [ -z ${OMP_NUM_THREADS+x} ]; then 
        echo "Variable OMP_NUM_THREADS is unset"
        exit 1
    fi

    if [ -n "$CONTAINER_PLT" ] ; then
        export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    fi

    return 0
}

# To ease PyTorch's FSDP hybrid sharding is fundamental that each process knows how many other processes are allocated on the same machine (i.e., the number of local GPUs), so that inter-node sharding and inter-node replication is handled correctly
# This is necessary since each process is run on only 1GPU, and some SLURM installation do not reset correctly the CUDA_VISIBLE_DEVICES variable
Reset_visible_devices () {
    if [ "${XFFL_EXECUTION}" = "true" ] ; then
        export VISIBLE_DEVICES=$( seq -s , 0 $(( LOCAL_WORLD_SIZE - 1 )) )
    elif command -v srun > /dev/null ; then # Check SLURM
        export VISIBLE_DEVICES=$( seq -s , 0 $(( SLURM_GPUS_PER_NODE - 1 )) ) # TODO: change SLURM_GPUS_PER_NODE for cloud environments
    fi

    return 0
}

# Default LLaMA variables - it's better to have them here than inside the llama-recipes code
LLaMA_default_env () {
	export TORCH_SHOW_CPP_STACKTRACES=1
	export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
	export TORCH_DISTRIBUTED_DEBUG="DETAIL"
	export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" # Not supported on Leonardo
	export TORCH_DISABLE_ADDR2LINE=1
	export NCCL_CROSS_NIC=1

    if [ -n "$CONTAINER_PLT" ] ; then
        export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES}"
        export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING}"
        export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG}"
        export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
        export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}TORCH_DISABLE_ADDR2LINE=${TORCH_DISABLE_ADDR2LINE}"
        export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}NCCL_CROSS_NIC=${NCCL_CROSS_NIC}"
    fi

}

# Check which GPU architecture is available on the current computing node
Gpu_detection () {
    # Check Nvidia GPU
    if command -v nvidia-smi > /dev/null ; then 
        export CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES	
        export GPU_FLAG="--nv"

        if [ -n "$CONTAINER_PLT" ] ; then
            export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        fi

        return 0
    fi 

    # Check AMD GPU
    if command -v rocm-smi > /dev/null ; then 
        export HIP_VISIBLE_DEVICES=$VISIBLE_DEVICES
        export ROCR_VISIBLE_DEVICES=$VISIBLE_DEVICES		
        export GPU_FLAG="--rocm"

        if [ -n "$CONTAINER_PLT" ] ; then
            export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
            export ENVIRONMENT="${ENVIRONMENT} ${PREFIX}ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES}"
        fi

        return 0
    fi

    echo "No GPU detected - falling back to CPU training"
    return 0
}

# Check which containerization software is available on the current computing node
Container_platform_detection () {
    unset PREFIX
    unset ENVIRONMENT
    unset CONTAINER_PLT

    # Check if `singularity` command exists
    if command -v singularity > /dev/null ; then 
        export CONTAINER_PLT="singularity"
        export PREFIX="SINGULARITYENV_"
        return 0
    fi 

    # Check if `apptainer` command exists
    if command -v apptainer > /dev/null ; then 
        export CONTAINER_PLT="apptainer"
        return 0
    fi 

    # Check if `docker` command exists
    if command -v docker > /dev/null ; then 
        export CONTAINER_PLT="docker"
        return 0
    fi 

    echo "No container platform detected - falling back to local environment"
    return 0
}