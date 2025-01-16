#!/bin/bash

# Leonardo
sbatch --qos boost_qos_dbg --nodes 1 --ntasks 4 --time=00:15:00 --output $(pwd)/worker/workspace/logs/llama_test.out --error $(pwd)/worker/workspace/logs/llama_test.err  --job-name llama-test $(pwd)/worker/scripts/leonardo.slurm -m /llama/worker/workspace/llama3.1-8b -t 1000 -v 512 -w leonardo_test -s $RANDOM

#Karolina
sbatch --partition qgpu_exp --nodes 1 --ntasks 8 --time=00:15:00 --output $(pwd)/worker/workspace/logs/llama_test.out --error $(pwd)/worker/workspace/logs/llama_test.err  --job-name llama-test $(pwd)/worker/scripts/karolina.slurm -m /llama/worker/workspace/llama3.1-8b -t 1000 -v 512 -w karolina_test -s $RANDOM

#MeluXina
sbatch --qos test --nodes 1 --ntasks 4 --time=00:15:00 --output $(pwd)/worker/workspace/logs/llama_test.out --error $(pwd)/worker/workspace/logs/llama_test.err  --job-name llama-test $(pwd)/worker/scripts/meluxina.slurm -m /llama/worker/workspace/llama3.1-8b -t 1000 -v 512 -w meluxina_test -s $RANDOM

#MareNostrum
sbatch --qos acc_debug --nodes 1 --ntasks 4 --time=00:15:00 --output $(pwd)/worker/workspace/logs/llama_test.out --error $(pwd)/worker/workspace/logs/llama_test.err  --job-name llama-test $(pwd)/worker/scripts/marenostrum.slurm -m /llama/worker/workspace/llama3.1-8b -t 1000 -v 512 -w marenostrum_test -s $RANDOM

#LUMI
sbatch --partition dev-g --nodes 1 --ntasks 8 --time=00:15:00 --output $(pwd)/worker/workspace/logs/llama_test.out --error $(pwd)/worker/workspace/logs/llama_test.err  --job-name llama-test $(pwd)/worker/scripts/lumi.slurm -m /llama/worker/workspace/llama3.1-8b -t 1000 -v 512 -w lumi_test -s $RANDOM