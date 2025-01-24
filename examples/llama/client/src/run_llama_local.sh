#!/bin/bash

export LOCAL_WORLD_SIZE=4
export WORLD_SIZE=4			 					
export GROUP_WORLD_SIZE=4				
export ROLE_WORLD_SIZE=4 						
export ROLE_NAME="default"

export MASTER_ADDR=localhost
export MASTER_PORT=29500


LOCAL_RANK=0 RANK=0 ROLE_RANK=0 GROUP_RANK=0 python training.py -dbg -m /leonardo_scratch/fast/uToID_bench/23_llama_sc24/worker/workspace/llama3.1-8b -d /leonardo_scratch/fast/uToID_bench/23_llama_sc24/datasets &
LOCAL_RANK=1 RANK=1 ROLE_RANK=1 GROUP_RANK=0 python training.py -dbg -m /leonardo_scratch/fast/uToID_bench/23_llama_sc24/worker/workspace/llama3.1-8b -d /leonardo_scratch/fast/uToID_bench/23_llama_sc24/datasets &
LOCAL_RANK=2 RANK=2 ROLE_RANK=2 GROUP_RANK=0 python training.py -dbg -m /leonardo_scratch/fast/uToID_bench/23_llama_sc24/worker/workspace/llama3.1-8b -d /leonardo_scratch/fast/uToID_bench/23_llama_sc24/datasets &
LOCAL_RANK=3 RANK=3 ROLE_RANK=3 GROUP_RANK=0 python training.py -dbg -m /leonardo_scratch/fast/uToID_bench/23_llama_sc24/worker/workspace/llama3.1-8b -d /leonardo_scratch/fast/uToID_bench/23_llama_sc24/datasets &

wait
