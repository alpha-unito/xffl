#!/bin/bash -ex

#SBATCH --job-name=xFFL
#SBATCH --output=xFFL.out
#SBATCH --error=xFFL.err
#SBATCH --nodes=2
#SBATCH --time=00:15:00
#SBATCH --account=ICT25_MHPC_0
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --ntasks-per-node=4			# GPUs x 1 node
#SBATCH --cpus-per-task=8			# cores x 1 node / GPUs x 1 node
#SBATCH --gpus-per-node=4
#SBATCH --exclusive

srun bash -c "{{ streamflow_command }}" 2>&1
# cd /leonardo_scratch/fast/uToID_bench/xffl/ && \
# source .venv/bin/activate && \
# xffl -dbg simulate ./examples/llm/client/src/training.py -f leonardo -p "${SLURM_GPUS_PER_NODE}" -n $(scontrol show hostnames "${SLURM_JOB_NODELIST}" | tr -s \\n " ")  -v .venv \
#     -args -m llama3.1-8b -d clean_mc4_it --seed 42 --subsampling 128 -t 1 -dbg -fs 4 -b 5
