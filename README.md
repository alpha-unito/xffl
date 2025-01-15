# Llama-SuperComputing2024

This repo keeps track of all the produced/modified code for the federated, large-scale execution of Llama models.

## First step: HPC User Day

### General information

Llama-2 can be obtained from the [META website](https://ai.meta.com/llama/) by clonining the Llama-2 [GitHub repository](https://github.com/facebookresearch/llama) and following the provided commands, modulo having obtained a valid download token from META.

This first round of implementation and experimentation takes into account the 7B Llama-2 model, trained on two different HPC infrastructures: [Leonardo](https://leonardo-supercomputer.cineca.eu) (Italy) and [Karolina](https://www.it4i.cz/en/infrastructure/karolina) (Czech Republic).
The datasets chosen for the two training are a [cleaned version of the MC4 Italian split corpus](https://huggingface.co/datasets/gsarti/clean_mc4_it) (clean_mc4_it) provided by [Gabriele Sarti](https://huggingface.co/gsarti) in its tiny version (10M documents for training and 12K documents for testing) for Leonardo, and a subset of the [MC4 Czech split](https://huggingface.co/datasets/mc4/viewer/cs) (mc4_cs) exactly matching the dimensions of the previous dataset.

The distributed execution of the various components of this project is handled by [StreamFlow](https://streamflow.di.unito.it), with the aggregation of the fine-tuned models produced by the supercomputer on a third cloud infrastructure.

The training of LLAMA-2 for Italians and Czechians relied on a prompt-tuning approach for an open-ended generation task. We fed a template “scrivi un seguente documento/Napište dokument:: {{text}}” with all the Italian and Czech documents included in the multilingual version of C4, computing the perplexity between the generated text and the document passed on the template.

### Technical details

The pre-trained Llama-2 model provided by HuggingFace comes in 16-bit precision and is then cast to 32-bit precision at runtime. When saved on disk in half-precision, it occupies ~13GB of memory. Each time a model is saved and loaded in this project, the half-precision format is assumed. Before using the model, it should be converted into the HuggingFace format, as specified in the [llama-recipes repository](https://github.com/facebookresearch/llama-recipes), the main codebase used for running these experiments.

The two datasets have already been tokenized offline on the C3S computational infrastructure to reduce execution times on the supercomputers. Each round of tokenization takes tens of computational hours on the CascadeLake nodes (it is a CPU-only computation). In their tokenization format, the Italian corpus weighs 102GB on disk, while the Czech one weighs 169 GB. Both datasets have been obtained from the HuggingFace repository, and the content of each entry's "text" field has been added as the answer to the prompt "Write a document:". The Llama tokenizer produces a 2048x3 embedding for each input document, independent of its size.
The tokenization process has been carried out through the `tokenizer.py` script.

SLURM handles the deployment on both Leonardo and Karolina and supports multi-node multi-GPU execution. The execution environment leverages the `torch run` utility to handle the complexity of the distributed training and `run` to deploy a torch run process on each of the booked computational nodes. The SLURM files used for the experiments are `slurm_leonardo.sh` and `slurm_karolina.sh`, and contain comments explaining part of the process and parameters. The `--use_fast_kernels` flags allow faster kernels with respect to the base one but require installing the `optimum` Python package. Furthermore, the sharding strategy has been set to `_HYBRID_SHARD_ZERO2`, which seems to offer better computational performance (more info [here](https://github.com/pytorch/pytorch/blob/release/2.2/torch/distributed/fsdp/api.py)), and no restriction has been set to the OMP_NUM_THREADS.

The aggregation step is carried out by the `aggregation.py` script, which averages the weights of the two provided networks and saves the result in half-precision in the same format as the HuggingFace pre-trained models. This code does not consider the amount of data on which the models were trained or the optimizer state.

Examples of commands for running the above functionalities are provided in `commands_examples`.

### Libraries modifications

Two libraries needed for the execution needed to be modified to support our experiments.

`Bits&Bytes` is a library for 8-bit quatisation of Deep Neural Networks. Despite not being actively used in this project, this library is deeply embedded in the llama-recipes code, and thus, it has not been removed. It can cause problems at runtime since it searches for paths related to the CUDA deployment. On Leonardo, the library follows a path in the Scratch file system without permission to access it and tries to remove it. Instead of gracefully handling this issue, Bits&Bytes produces an exception and makes the whole computation fail. This is fixed by ignoring this kind of exception (`bitsandbytes/bitsandbytes/cuda_setup/main.py,` function `remove_non_existent_dirs`).

`Llama-recipes` is a repository containing ready-made code to execute Llama-2. Despite being relatively easy to use, some modifications have been applied to obtain the desired behavior. 

Firstly, the saving functionalities of a distributed model (FSDP, Fully Sharded Data Parallelism) did not match the expected behavior. By modifying the checkpoint handling logic, this library now handles the saving of a distributed model equivalently as a sequential pre-trained model: the different shards are collected, converted into half-precision, and saved in the command-line specified folder in the same format as HuggingFace pre-trained models (`llama-recipes/src/llama_recipes/config/fsdp.py`, `checkpoint_type: StateDictType = StateDictType.FULL_STATE_DICT`, and `llama-recipes/src/llama_recipes/model_checkpointing/checkpoint_handler.py`, function `save_model_checkpoint`).

Secondly, the logic for loading the pre-processed, tokenized datasets has been implemented. This required to apply easily interpretable modifications to multiple files (`llama-recipes/src/llama_recipes/config/datasets.py`, `llama-recipes/src/llama_recipes/datasets/__init__.py`, `llama-recipes/src/llama_recipes/utils/dataset_utils.py`) and the creation of two new classes specifically designed for loading a specific amount of data (hard-coded in the classes themselves via `TRAIN_SAMPLES` and `VAL_SAMPLES`) from the provided tokenized datasets (`llama-recipes/src/llama_recipes/datasets/clean_mc4_it_dataset.py`, `llama-recipes/src/llama_recipes/datasets/mc4_cs_dataset.py`).

Thirdly, since the `--use_fast_kernels` converts the model structure into the BetterTransformer format, the logic for saving the sharded model broke again since the reverse conversion of the model to the Transformer format crashes. To this end, once the fast kernels are chosen, the model is copied before being converted so that when it comes the time to save the sharded model to disk, the old model can still be used as a reference for the original model structure. This quick fix required to propagate the copy of the original model through all the methods stack (`llama-recipes/src/llama_recipes/finetuning.py`, `llama-recipes/src/llama_recipes/utils/train_utils.py`, `llama-recipes/src/llama_recipes/model_checkpointing/checkpoint_handler.py`, reference the `model_old` variable).

Fourthly, the `llama-recipes/src/llama_recipes/finetuning.py` loads the tokenizer from the same folder as the Llama model. Since the saved checkpoints do not also contain the tokenizer model, all the tokenizer-related files contained in the original `llama-2-7b` folder have been moved to the `tokenizer` folder, and an additional parameter (`--tokenizer`) has been added to the script to load the tokenizer from the provided path (`llama-recipes/src/llama_recipes/finetuning.py`, `llama-recipes/src/llama_recipes/configs/training.py`).

Fifthly, different sharding strategies are required by different infrastructures. Due to the larger GPU RAM, Leonardo can bear the most advanced and efficient sharding technique available, `_HYBRID_SHARD_ZERO2`, while Karolina, having less GPU RAM, can bear the second most efficient sharding strategy, `HYBRID_SHARD`. Such values should be inserted in the FSDP configuration file (`llama-recipes/src/llama_recipes/configs/fsdp.py`) and are not specified a priori to avoid using the wrong sharding strategy. Furthermore, these sharding strategies reduce the inter-node communication overhead, effectively improving scaling performance over multiple nodes.

Furthermore, the `padding` batching strategy is used instead of the standard `packing` one to avoid additional preprocessing, and the GPU selected by `llama-recipes/src/llama_recipes/finetuning.py` and `llama-recipes/src/llama_recipes/utils/train_utils.py` is changed from `local_rank` to `torch.cuda.current_device()` to accomodate such cases in which there are multiple processes on the same node but, due to SLURM environemt settings, each of them sees only one GPU indexed as `0`. `torch.cuda.set_device()` is initialised accordingly.

### AMD-specific modifications

A couple of quick fixes are needed to run Llama-2 on AMD-based infrastructures (e.g., Pawsey's Setonix). The practitioner should consider that such platforms use the ROCM software stack (instead of CUDA), and thus, some Python packages designed assuming a CUDA backend could require some work to be adapted to this different backend.

Notably, the 8-bit quantization library Bits&Bytes still does not support the ROCm backend at the time of writing, requiring moving to an unofficial porting of Bits&Bytes supporting the ROCm environment: [Bits&Bytes-ROCm](https://git.ecker.tech/mrq/bitsandbytes-rocm). This software has been compiled (it requires the shader ISA code of the underlying AMD GPU, for example, gfx90a for the AMD Radeon Instinct MI250X, ROCm version 5.6, nvcc, and the HiPcc compiler) and substituted to the standard pip version of Bits&Bytes, effectively allowing the training process.

Furthermore, the `llama-recipes` code implicitly assumes an underlying NVIDIA/CUDA environment and, thus, explicitly checks for the installed CUDA version to decide if brain float support is available or not, thus implicitly disabling it on AMD/ROCm environments and forcing FP32 precision. To avoid this, the checks on the cuda version present in `llama-recipes` have been commented out (`llama-recipes/src/llama_recipes/utils/train_utils.py`, method `get_policies`), effectively enabling brain float 16bit mixed precision training also on AMD GPUs.

### Bypassing torchrun

Torchrun (also known as Elastic Launch) is a utility made available by PyTorch allowing to running multi-process PyTorch code without having the user worry about the actual distributed run details, such as the rank assignment, the process groups creation in case of multi-node multi-gpu training, and so on. Furthermore, it provides the so-called "elasticity", a soft form of fault tolerance, allowing the graceful handling of non-responding PyTorch processes, eventually restarting them. Unfortunately, this utility relies on a client-server architecture based on TCP communication, which can cause troubles in HPC environments where the number of TCP connections can be limited. Since our use case does not need fault-tolerance properties, we eliminated the torch run dependency by setting the needed environmental variable ourselves. We first investigated how torchrun sets them (`torchrun_env_var`) and then replicated the behavior through a static bash script, not relying on TCP communication (`launch.slurm`, which in turn calls `torchrun.sh`, can be found in the `script_leonardo` or `script_karolina` folder). This approch effectively allow us to scale over the 128 nodes limit we encouterend on Leonardo, exploiting the InfiniBand interconnection network.

### Minor fixes

HuggingFace `datasets` libraries connects to the network bu default. This fact can be annoing when working on HPC infrastructures, where nodes do not have internet connection. From a practical perspective, this behaviour can create problem when trying to access datasets. The `HF_DATASETS_OFFLINE` environment variable can be set to `1` to bypass internet requestes, thus not wasting time waiting for the timeovers. Note that, if working offline, the datasets should already been downloaded and available. If that is the case, loading a dataset simply requires to insert its local path in the `load_dataset` function in the `llama_recipes/datasets` dataset's file instead of the dataset name.

Some minor warning have been fixed to obtain a cleaner execution. In `transformers/utils/generic.py:441` and `transformers/utils/generic.py:441` the deprecated `torch.utils._pytree._register_pytree_node` has ben substituted with `torch.utils._pytree.register_pytree_node`; also, the deprecated `prepare_model_for_int8_training` in `llama_recipes/finetuning.py` has been substituted with `prepare_model_for_kbit_training`. In `llama_recipes/utils/memory_utils.py` the deprecated `torch.cuda.reset_max_memory_allocated` has been sustituted with `torch.cuda.reset_peak_memory_stats`.

### Singularity containers

Singularity containers are created to allow protability and reproducibility of the experiments.
A general Singularity definition file (`sc_container.def`) is provided as recipe for all containers and platforms included in the experiments.
The NVIDIA official optimised `nvcr.io/nvidia/pytorch:24.02-py3` is used as base image for the Intel-NVIDIA architectures (the theoretically more Leonardo-compatible `nvcr.io/nvidia/pytorch:23.07-py3` produced CUDA-out-of-memory errors), while `rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1` is exploited on AMD-ROCM architectures.
To optimise the computational performance on LUMI, the LUMI's official Singularity image for PyTorch (`lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.0-dockerhash-f72ddd8ef883.sif`) is used after being updated with the Python packages required by LLAMA described above.
It should be noted that LUMI's official supported ROCM version is 5.2, which is indeed quite old.
By using the `module` software, the ROCM version used in the experiments has been raised to 5.6.1.
More specifically, the base image has been moved from LUMI to a Intel VM where it is possible to build Singularity containers, converted to a sandbox (`--sandbox` Singularity option), opened as writable (`--writable` singularity option), updated with the required software, re-converted to a standard Singularity image and moved back to LUMI.
In this case, Bits&Bytes has been manually compiled on LUMI, then moved to the VM and included into the final container to avoid microarchitecture-related compatibility issues.
The containers are run with the `--nv` option on NVIDIA clusters and `--rocm` option on ROCM clusters to allow GPU usage.
