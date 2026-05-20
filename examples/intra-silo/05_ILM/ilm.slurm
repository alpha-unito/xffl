#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --account=gmittone
#SBATCH --partition=gracehopper
#SBATCH --gpus-per-node=1
#SBATCH --tasks-per-node=1
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gianluca.mittone@unito.it

cd /beegfs/home/gmittone/ILM/xffl/examples/intra-silo/05_ILM
source /beegfs/home/gmittone/ILM/xffl/.venv/bin/activate

MODEL_NAME=${MODEL_NAME} ILM_TYPE=${ILM_TYPE} ILM_LANGUAGE_A=${ILM_LANGUAGE_A} ILM_LANGUAGE_B=${ILM_LANGUAGE_B} ILM_TOKENIZER=${ILM_TOKENIZER} xffl exec training.py config_qwen.py > /beegfs/home/gmittone/ILM/ILM/output/logs/${MODEL_NAME}_${ILM_TYPE}_${ILM_LANGUAGE_A}_${ILM_LANGUAGE_B}_${ILM_TOKENIZER}.log
