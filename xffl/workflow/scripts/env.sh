#!/bin/bash

# The PyTorch's FSDP runtime requires the following environment variables to be correctly set to operate correctly.
# These environment variables are very similar to those used by other distributed systems (i.e., MPI)
# We extract this information from the SLURM runtime environment through srun, but an equivalent setting is obtained through mpirun (mpi equivalent variables are indicated in comments)
Derive_env_from_SLURM () {
	if (( SLURM_NTASKS_PER_NODE >= SLURM_NTASKS )); then		# SLURM_GPUS_ON_NODE, OMPI_COMM_WORLD_LOCAL_SIZE
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
}

# Due to PyTorch's aggressive thread policy OMP_NUM_THREADS should be manually set to the number of actually available cores (by default PyTorch would spawn a thread for each processor's core)
Limit_PyTorch_threads () {
	export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
}

# To agevolate PyTorch's FSDP hybrid sharding is fundamental that each process knows how many other processes are allocated on the same machine (i.e., the number of local GPUs), so that inter-node sharding and inter-node replication is handled correctly
# This is necessary since each process is run on only 1GPU, and some SLURM installation do not reset correctly the CUDA_VISIBLE_DEVICES variable
Reset_visible_devices () {
	VISIBLE_DEVICES=$( seq -s , 0 $(( SLURM_GPUS_PER_NODE - 1 )) ) # todo: change SLURM_GPUS_PER_NODE for cloud environments
	export VISIBLE_DEVICES
}

# Default LLaMA variables - it's better to have them here than inside the llama-recipes code
LLaMA_default_env () {
	export TORCH_SHOW_CPP_STACKTRACES=1
	export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
	export TORCH_DISTRIBUTED_DEBUG="DETAIL"
	export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
	export TORCH_DISABLE_ADDR2LINE=1
	export NCCL_CROSS_NIC=1
}

# Check which GPU architecture is available on the current computing node
Gpu_detection () {
    # Check Nvidia GPU
    if nvidia-smi > /dev/null ; then 
        export CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES		
        GPU_FLAG="--nv" 
        return 0
    fi 

    # Check AMD GPU
    if rocm-smi > /dev/null ; then 
        export HIP_VISIBLE_DEVICES=$VISIBLE_DEVICES
        export ROCR_VISIBLE_DEVICES=$VISIBLE_DEVICES		
        GPU_FLAG="--rocm" 
        return 0
    fi 

    echo "[RANK ${RANK}] GPU detection FAILED"
    exit 1
}

# Check which containerization software is available on the current computing node
Container_platform_detection () {
    # Check if `singularity` command exists
    if singularity --version > /dev/null ; then 
        CONTAINER_PLT="singularity"
        return 0
    fi 

    # Check if `apptainer` command exists
    if apptainer --version > /dev/null ; then 
        CONTAINER_PLT="apptainer" 
        return 0
    fi 

    # Check if `docker` command exists
    if docker --version > /dev/null ; then 
        CONTAINER_PLT="docker"
        return 0
    fi 

    echo "[RANK ${RANK}] Container Platform detection FAILED"
    exit 1
}