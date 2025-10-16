#!/bin/bash -ex

#SBATCH --chdir=/leonardo_scratch/fast/uToID_bench/xffl/examples/simulation/03_LLM/
#SBATCH --mail-user=gianluca.mittone@unito.it
#SBATCH --mail-type=ALL
#SBATCH --time=00:15:00
#SBATCH --account=ICT25_MHPC_0
#SBATCH --partition=boost_usr_prod
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --nodes=16
#SBATCH --qos=normal
#SBATCH --job-name=xffl-test
#SBATCH --output=xffl-test.out
#SBATCH --error=xffl-test.err

xffl --debug exec training.py config.py --image /leonardo_scratch/fast/EUHPC_B27_058/image.sif
