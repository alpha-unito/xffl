#!/bin/bash -ex

#SBATCH --job-name=xFFL
#SBATCH --error=xFFL.err
#SBATCH --output=xFFL.out
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --time=00:15:00
#SBATCH --account=gmittone
#SBATCH --partition=gracehopper
#SBATCH --gpus-per-node=1
#SBATCH --tasks-per-node=1
#SBATCH --exclusive

cd /beegfs/home/gmittone/xffl && \
source .venv-gh/bin/activate && \
spack load python@3.12.1 cuda@12.3.2 cudnn@8.9.5.30-12 && \
pip install . 1>&2 && \
PYTHONUNBUFFERED=1 xffl -dbg simulate ./examples/llm/client/src/training.py -f hpc4ai-gh -p ${SLURM_GPUS_PER_NODE} -n $(scontrol show hostnames $SLURM_JOB_NODELIST | tr -s \\n " ")  -v .venv-gh \
    -args -m llama3.1-8b -d clean_mc4_it --seed 42 --subsampling 128 -dbg -ws 8

