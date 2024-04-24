#!/bin/bash

echo "*** xFFL - Coordinator setup ***"
echo "Assuming the local directory as working directory"

echo "Workspace creation..."
mkdir coordinator
cd coordinator

echo "Python xffl virtual environment creation..."
python -m venv xffl
source xffl/bin/activate

echo "Installing xFFL requirements..."
pip install -r ../aggregator/requirements.txt

echo "Installing StreamFlow..."
git clone --depth 1 https://github.com/alpha-unito/streamflow.git
cd streamflow
pip install -r requirements.txt .
cd ..

echo "Installing LLaMA..."
git clone --depth 1 https://github.com/meta-llama/llama.git
cd llama
pip install -e .

echo "LLaMA download, META url required..."
bash download.sh
mv tokenizer.model llama-2-7b/
cd ..

echo "Installing transformers..."
git clone --depth 1 https://github.com/huggingface/transformers.git
cd transformers
pip install protobuf

echo "LLaMA to HuggingFace convertion..."
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
   --input_dir ../llama/llama-2-7b --model_size 7B --output_dir ../llama-2-7b
cd ..

echo "Removing downloaded repositories..."
rm -rf llama transformers streamflow

echo "*** xFFL - Coordinator setup - Done!***"
