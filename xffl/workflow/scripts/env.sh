#!/bin/bash

# The PyTorch's FSDP runtime requires the following environment variables to be correctly set to operate correctly.
# These environment variables are very similar to those used by other distributed systems (i.e., MPI)
# We extract this information from the SLURM runtime environment through srun, but an equivalent setting is obtained through mpirun (mpi equivalent variables are indicated in comments)
Derive_env () {

    if [ "${XFFL_FACILITY}" = "local" ] ; then
        export LOCAL_WORLD_SIZE=$XFFL_WORLD_SIZE
        export WORLD_SIZE=$XFFL_WORLD_SIZE		 					
        export GROUP_WORLD_SIZE=$XFFL_WORLD_SIZE				
        export ROLE_WORLD_SIZE=$XFFL_WORLD_SIZE 						
        export ROLE_NAME="default"

        # RANKs are set in the facilitator

        export MASTER_ADDR=localhost
        export MASTER_PORT=29500

        export GROUP_RANK=0
    elif command -v srun > /dev/null ; then # Check SLURM
        if (( SLURM_NTASKS_PER_NODE >= SLURM_NTASKS )); then		# SLURM_GPUS_ON_NODE, OMPI_COMM_WORLD_LOCAL_SIZE
            export LOCAL_WORLD_SIZE=$SLURM_NTASKS
        else
            export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE
        fi
        export WORLD_SIZE=$SLURM_NTASKS			 					# OMPI_COMM_WORLD_SIZE
        export GROUP_WORLD_SIZE=$SLURM_JOB_NUM_NODES				# OMPI_MCA_orte_num_nodes 
        export ROLE_WORLD_SIZE=$SLURM_NTASKS 						# OMPI_COMM_WORLD_SIZE
        export ROLE_NAME="default"

        export LOCAL_RANK=$SLURM_LOCALID 							# OMPI_COMM_WORLD_LOCAL_RANK
        export RANK=$SLURM_PROCID 									# OMPI_COMM_WORLD_RANK
        export ROLE_RANK=$RANK										# $OMPI_COMM_WORLD_RANK
        export GROUP_RANK=$(( RANK / SLURM_NTASKS_PER_NODE ))
        

        export MASTER_ADDR=$SLURM_SRUN_COMM_HOST
        export MASTER_PORT=29500
    fi

    return 0
}

# Due to PyTorch's aggressive thread policy OMP_NUM_THREADS should be manually set to the number of actually available cores (by default PyTorch would spawn a thread for each processor's core)
Limit_PyTorch_threads () {
    if [ "${XFFL_FACILITY}" = "local" ] ; then
	    export OMP_NUM_THREADS=$(( $(nproc) / WORLD_SIZE ))
    elif command -v srun > /dev/null ; then # Check SLURM
        export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    fi

    return 0
}

# To agevolate PyTorch's FSDP hybrid sharding is fundamental that each process knows how many other processes are allocated on the same machine (i.e., the number of local GPUs), so that inter-node sharding and inter-node replication is handled correctly
# This is necessary since each process is run on only 1GPU, and some SLURM installation do not reset correctly the CUDA_VISIBLE_DEVICES variable
Reset_visible_devices () {
    if [ "${XFFL_FACILITY}" = "local" ] ; then
        VISIBLE_DEVICES=$( seq -s , 0 $(( 4 - 1 )) )
    elif command -v srun > /dev/null ; then # Check SLURM
        VISIBLE_DEVICES=$( seq -s , 0 $(( SLURM_GPUS_PER_NODE - 1 )) ) # TODO: change SLURM_GPUS_PER_NODE for cloud environments
    fi
    export VISIBLE_DEVICES

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
}

# Check which GPU architecture is available on the current computing node
Gpu_detection () {
    # Check Nvidia GPU
    if command -v nvidia-smi > /dev/null ; then 
        export CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES		
        GPU_FLAG="--nv"
        return 0
    fi 

    # Check AMD GPU
    if command -v rocm-smi > /dev/null ; then 
        export HIP_VISIBLE_DEVICES=$VISIBLE_DEVICES
        export ROCR_VISIBLE_DEVICES=$VISIBLE_DEVICES		
        GPU_FLAG="--rocm"
        return 0
    fi 

    echo "No GPU detected - falling back to CPU training"
    return 0
}

# Check which containerization software is available on the current computing node
Container_platform_detection () {
    # Check if `singularity` command exists
    if command -v singularity > /dev/null ; then 
        CONTAINER_PLT="singularity"
        return 0
    fi 

    # Check if `apptainer` command exists
    if command -v apptainer > /dev/null ; then 
        CONTAINER_PLT="apptainer" 
        return 0
    fi 

    # Check if `docker` command exists
    if command -v docker > /dev/null ; then 
        CONTAINER_PLT="docker"
        return 0
    fi 

    echo "No container platform detected - falling back to local environment"
    return 0
}