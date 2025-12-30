#!/bin/bash -ex

ulimit -n 131072
module load cuda/12.2 nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22

# LEONARDO specific environment variables
export XFFL_LOCAL_TMPDIR=${TMPDIR}
export PYTHONUNBUFFERED=1

# export NCCL_DEBUG=INFO
# export TORCH_NCCL_TRACE_BUFFER_SIZE=1000
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=SYS
export NCCL_NET_GDR_LEVEL=2
