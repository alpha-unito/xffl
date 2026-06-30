#!/bin/bash

ulimit -n 131072

module load gcc/12.2.0 cuda/12.6 python/3.11.7

export PYTHONUNBUFFERED=1

# export NCCL_DEBUG=INFO
# export TORCH_NCCL_TRACE_BUFFER_SIZE=1000
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=SYS
export NCCL_NET_GDR_LEVEL=2
