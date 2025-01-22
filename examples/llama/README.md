# HPC federated training of LLaMA 3.1 - 8B

[HPC federated training of LLaMA](https://hpc4ai.unito.it/hpc-federation/)



## Setup

### Aggregator
```bash
singularity build aggregator.sif aggregator.def
```

```bash
pip install llama-stack

git clone --depth 1 https://github.com/huggingface/transformers.git
llama model download --source meta --model-id Llama3.1-8B
python transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir .llama/checkpoints/Llama3.1-8B/ --model_size 8B --output_dir llama3.1-8b --llama_version 3.1
```

### Client
```bash
singularity build client.sif client.def

python llama3.1-8b ../tokenization/tokenizer.py gsarti/clean_mc4_it tiny 0
mv gsarti_clean_mc4_it_* ../../datasets/
cd ../../datasets/ || exit
mv gsarti_clean_mc4_it_train/ clean_mc4_it_train.hf/
mv gsarti_clean_mc4_it_val/ clean_mc4_it_val.hf
```