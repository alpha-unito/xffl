#!/bin/bash

echo "*** xFFL - Facility setup ***"
echo "Assuming the local directory as working directory"

echo "Workspace creation..."
mkdir facility
cd facility

echo "Python xffl virtual environment creation..."
python -m venv xffl
source xffl/bin/activate

echo "Installing xFFL requirements..."
cd ../libraries/llama-recipes/
pip install -r ../../requirements.txt .

echo "Dataset download and tokenisation..."
cd ../../dataset
python tokenizer.py gsarti/clean_mc4_it tiny 0
# python tokenizer.py mc4 cs 0

echo "*** xFFL - Facility setup - Done!***"
