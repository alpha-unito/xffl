#!/bin/bash

echo "*** xFFL - Worker setup ***"
echo "Assuming the local directory as working directory"

echo "Workspace creation..."
mkdir -p workspace
mkdir -p ../datasets
cd workspace

echo "Singularity container creation and start..."
singularity build worker.sif ../worker.def
singularity shell --mount type=bind,src=$(pwd),dst=/workspace/ --home /workspace/ worker.sif

echo "LLaMA repositories download..."
git clone --depth 1 https://github.com/huggingface/transformers.git

echo "LLaMA 3.1 8B & 70B model download and conversion..."
llama model download --source meta --model-id Llama3.1-8B
python transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir .llama/checkpoints/Llama3.1-8B/ --model_size 8B --output_dir llama3.1-8b --llama_version 3.1
llama model download --source meta --model-id Llama3.1-70B
python transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir Llama3.1-70B/ --model_size 70B --output_dir llama3.1-70b --llama_version 3.1

echo "Mixtral-8x7B-v0.1 model download..."
huggingface-cli login
huggingface-cli download --local-dir mixtral-8x7b-v0.1 mistralai/Mixtral-8x7B-v0.1

echo "Dataset download and tokenisation..."
python llama3.1-8b ../tokenization/tokenizer.py gsarti/clean_mc4_it tiny 0
mv gsarti_clean_mc4_it_* ../../datasets/
cd ../../datasets/
mv gsarti_clean_mc4_it_train/ clean_mc4_it_train.hf/
mv gsarti_clean_mc4_it_val/ clean_mc4_it_val.hf
exit

echo "*** xFFL - Worker setup - Done! ***"
