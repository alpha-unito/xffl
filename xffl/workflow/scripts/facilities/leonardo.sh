#!/bin/bash
killall python

module load cuda/12.1
# LEONARDO specific environment variables
export XFFL_LOCAL_TMPDIR=${TMPDIR}
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000