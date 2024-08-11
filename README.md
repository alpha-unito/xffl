# Cross-Facility Federated Learning

## EuroHPC User Day 2023 - Reproducibility

This file describes all the steps necessary to reproduce the experiments reported by Iacopo Colonnelli during his talk "Cross-Facility Federated Learning" (xFFL) at the EuroHPC user day 2023 and subsequently published in the [conference proceedings](https://dx.doi.org/10.1016/j.procs.2024.07.003). If you want to cite XFFL, please use the reference below:

```bibtex
@article{24:eurohpc:xffl,
    title = {Cross-Facility Federated Learning},
    author = {Iacopo Colonnelli and
              Robert Birke and
              Giulio Malenza and
              Gianluca Mittone and
              Alberto Mulone and
              Jeroen Galjaard and
              Lydia Y. Chen and
              Sanzio Bassini and
              Gabriella Scipione and
              Jan Martinovi\v{c} and
              Vit Vondr\'{a}k and
              Marco Aldinucci},
    doi = {10.1016/j.procs.2024.07.003},
    issn = {1877-0509},
    year = {2024},
    booktitle = {Proceedings of the First EuroHPC user day},
    journal = {Procedia Computer Science},
    volume = {240},
    pages = {3--12},
    publisher = {Elsevier},
    address = {Amsterdam, Netherlands},
    location = {Bruxelles, Belgium},
}
```

## Reproducibility disclaimer

### Exact reproducibility

The presented results were obtained by experiments run on the Leonardo and Karolina supercomputers in December 2023.
Infrastructure and software changes applied to the two mentioned supercomputing facilities can potentially lead to different outcomes; please take that into account when reproducing the reported experiments.
Furthermore, due to their scale, our experiments were run only once; many unpredictable factors (e.g., queue times, HPC current utilisation, intra- and inter-HPC network bandwidth available, et similia) can lead to potentially different outcomes.

The proposed experiments are reproducible even if the interested reader cannot access the Leonardo and Karolina supercomputing centres.
The provided scripts and instructions are compatible with any ssh-reachable HPC computing infrastructure.
Furthermore, the scripts are natively compatible with many queue management systems (e.g., SLURM, PBS, HyperQueue, Flux, et similia), easily allowing reproducibility tests on a wide range of computing infrastructure.

Finally, the StreamFlow WMS needs a third computing facility to run (named coordinator), which is not an HPC system.
Such a facility should be connected to the internet and capable of reaching the HPC infrastructure.
In the proposed experiments, StreamFlow is hosted on a cloud VM physically located near the Leonardo supercomputing system, thus offering fast internet data transfer performance between the two machines.

However, these considerations do not hinder the validity of the presented work; instead, they constitute an inherent demonstration of the complexity of cross-facility deployments, proving even more the necessity for sophisticated software tools to handle such large-scale distributed computations.

### Software requirements

python>=3.8\
streamflow>=0.2.0.dev10\
torch>=2.2.0\
llama\
transformers>=4.34.1

## Reproduce the experiments

### Step 1: HPC setup

The following setup steps must be run on each HPC infrastructure participating in the federation.

#### Automatic

Just clone this repository and run:

```bash
bash scripts/facility_setup.sh
```

This will download the full gsart/clean_mc_it dataset in its tiny version. To modify the default dataset parameters, edit the script. That is it!

#### Manual

As a first step, proceed to create and activate a new Python virtual environment; venv and Conda are usable interchangeably:

```bash
mkdir facility
cd facility
python -m venv xffl
source xffl/bin/activate
```

Then, proceed to install xFFL Python requirements and the modified llama-recipes library:

```bash
cd ../libraries/llama-recipes/
pip install -r ../../requirements.txt .
```

Optionally, the modified bitsandbytes library found in the `libraries/` folder can be built from scratch and installed in case of AMD processors.
<!--- 
cd ../bitsandbytes
pip install . 
--->

Finally, download the desired dataset and tokenise it:

```bash
cd ../../dataset
python tokenizer.py gsarti/clean_mc4_it tiny 0 # Leonardo
python tokenizer.py mc4 cs 0 # Karolina
```

The first command-line parameter indicates the HuggingFace data to download, the second inversion, and the third the number of samples to tokenise to handle the experiments' length (0 equals using the whole dataset).

### Step 2: Coordinator setup

The second step in reproducing the experiments is preparing the coordinator. This can be done manually by following the reported commands step-by-step or by running the `coordinator_setup.sh` script on the coordinator machine. Make sure to have a Python interpreter and internet connection available before starting!

#### Automatic

The only prerequisite to auto-install the coordinator software is submitting a Llama 2 access request on the [official Meta website](https://llama.meta.com/llama-downloads/) and receiving a unique custom access URL via email.
Once that is done, clone this repository and run:

```bash
bash scripts/coordinator_setup.sh
```

prompt the META URL when asked, and download the 7B model. Done!

#### Manual

As a first step, after cloning this repository, proceed to create and activate a new Python virtual environment; venv and Conda are usable interchangeably:

```bash
mkdir coordinator
cd coordinator
python -m venv xffl
source xffl/bin/activate
```

Install then the Python xFFL requirements:

```bash
pip install -r ../aggregator/requirements.txt
```

Proceed then to install the StreamFlow WMS from source to obtain the latest features necessary to support the cross-facility scenario (the pip-available StreamFlow package is currently outdated):

```bash
git clone --depth 1 https://github.com/alpha-unito/streamflow.git
cd streamflow
pip install -r requirements.txt .
cd ..
```

Next, download the LLaMA-2 7B model.
This step requires submitting a Llama 2 & Llama Chat access request on the [official Meta website](https://llama.meta.com/llama-downloads/) and receiving a unique custom access URL via email.
Meanwhile, it is necessary to clone and install the official META llama repository:

```bash
git clone --depth 1 https://github.com/meta-llama/llama.git
cd llama
pip install -e .
bash download.sh # The custom access URL is required here
mv tokeniser.model llama-2-7b/ # Required for the following model conversion
cd ..
```

Finally, convert the LLaMA-2 7B model to the HuggingFace format:

```bash
git clone --depth 1 https://github.com/huggingface/transformers.git
cd transformers
pip install protobuf
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
 --input_dir ../llama/llama-2-7b --model_size 7B --output_dir ../llama-2-7b
cd ..
```

Optionally, remove all the downloaded repositories:

```bash
rm -rf llama transformers streamflow
```

The coordinator node is ready to deploy the xFFL workload on the HPC infrastructures.

### Step 3: Experiments launching

Before launching the experiments, it is necessary to configure the StreamFlow file, which can be found in the `workflow/streamflow.yml` path. The user can define the preferred HPC infrastructures in the `deployments` section. Currently, the Leonardo and Karolina facilities are configured with the author accounts; thus, some changes are necessary. The user must eventually supply certificates or a private SSH key, username and working directory path. Moreover, the paths for each input data, `train_script_*`, `dataset_path_*`, and `tokeniser_*`, must be changed. In general, these paths must refer to the remote location of this repository inside the HPC facility (the directory created in Step 1). It is enough to fix the paths inside the `streamflow.yml` for the execution following this user guide. However, the user can also change the `cwl/config.yml` to customise the input data paths. In the default configuration, the `cwl/config.yml` has the input paths of this repository; thus, the `tokeniser` and `dataset` inputs are inside the `dataset` directory, the `train_script` input is inside the `libraries` directory and so on.

Each HPC facility has its directives on submitting a job. For this reason, the user must define a bash script template to define all these directives. In the repository, it is possible to find the templates to submit on Leonardo and Karolina in the `templates/cineca/leonardo.sh` and `templates/karolina/karo.sh` paths. These templates are already referred to in the streamflow.yml file. An important directive to change is `#SBATCH --account=[PROJECT_NAME]`. The user must write their project name, which has available computational hours. Another essential directive to change can be the number of nodes, e.g. `#SBATCH --nodes=128`, to improve the training model time. In this case, beyond the nodes directive, it is necessary to fix other directives like ntasks, ntasks_per_node, and so on.
Finally, the workflow execution is ready. The following command starts the run:

```bash
cd workflow
streamflow run streamflow.yml
```

It is suggested to execute the command in background, using commands such as `screen` or `nohup`.

## Experiments description

### General information

Llama-2 can be obtained from the [META website](https://ai.meta.com/llama/) by clonining the Llama-2 [GitHub repository](https://github.com/facebookresearch/llama) and following the provided commands, modulo having obtained a valid download token from META.

This first round of implementation and experimentation takes into account the 7B Llama-2 model, trained on two different HPC infrastructures: [Leonardo](https://leonardo-supercomputer.cineca.eu) (Italy) and [Karolina](https://www.it4i.cz/en/infrastructure/karolina) (Czech Republic).
The datasets chosen for the two training are a [cleaned version of the MC4 Italian split corpus](https://huggingface.co/datasets/gsarti/clean_mc4_it) (clean_mc4_it) provided by [Gabriele Sarti](https://huggingface.co/gsarti) in its tiny version (10M documents for training and 12K documents for testing) for Leonardo, and a subset of the [MC4 Czech split](https://huggingface.co/datasets/mc4/viewer/cs) (mc4_cs) exactly matching the dimensions of the previous dataset.

The distributed execution of this project's various components is handled by [StreamFlow](https://streamflow.di.unito.it), which aggregates the fine-tuned models produced by the supercomputer on a third cloud infrastructure.

The training of LLAMA-2 for Italians and Czechians relied on a prompt-tuning approach for an open-ended generation task. We fed a template "scrivi un seguente documento/Napište dokument:: {{text}}" with all the Italian and Czech documents included in the multilingual version of C4, computing the perplexity between the generated text and the document passed on the template.

### Technical details

The pre-trained Llama-2 model provided by HuggingFace comes in 16-bit precision and is then cast to 32-bit precision at runtime. When saved on disk in half-precision, it occupies ~13GB of memory. Each time a model is saved and loaded in this project, the half-precision format is assumed. Before using the model, it should be converted into the HuggingFace format, as specified in the [llama-recipes repository](https://github.com/facebookresearch/llama-recipes), the main codebase used for running these experiments.

The two datasets have already been tokenised offline on the C3S computational infrastructure to reduce execution times on the supercomputers. Each round of tokenisation takes tens of computational hours on the CascadeLake nodes (it is a CPU-only computation). In their tokenisation format, the Italian corpus weighs 102GB on disk, while the Czech one weighs 169 GB. Both datasets have been obtained from the HuggingFace repository, and the content of each entry's "text" field has been added as the answer to the prompt "Write a document:". The Llama tokeniser produces a 2048x3 embedding for each input document, independent of its size.
The tokenisation process has been carried out through the `tokenizer.py` script.

SLURM handles the deployment on both Leonardo and Karolina and supports multi-node multi-GPU execution. The execution environment leverages the `torch run` utility to handle the complexity of the distributed training and `run` to deploy a torch run process on each of the booked computational nodes. The SLURM files used for the experiments are `slurm_leonardo.sh` and `slurm_karolina.sh`, and contain comments explaining part of the process and parameters. The `--use_fast_kernels` flags allow faster kernels with respect to the base one but require installing the `optimum` Python package. Furthermore, the sharding strategy has been set to `_HYBRID_SHARD_ZERO2`, which seems to offer better computational performance (more info [here](https://github.com/pytorch/pytorch/blob/release/2.2/torch/distributed/fsdp/api.py)), and no restriction has been set to the OMP_NUM_THREADS.

The aggregation step is carried out by the `aggregation.py` script, which averages the weights of the two provided networks and saves the result in half-precision in the same format as the HuggingFace pre-trained models. This code does not consider the amount of data on which the models were trained or the optimiser state.

Examples of commands for running the above functionalities are provided in `commands_examples`.

### Libraries modifications

Two libraries needed for the execution needed to be modified to support our experiments.

`Bits&Bytes` is a library for 8-bit quatisation of Deep Neural Networks. Despite not being actively used in this project, this library is deeply embedded in the llama-recipes code, and thus, it has not been removed. It can cause problems at runtime since it searches for paths related to the CUDA deployment. On Leonardo, the library follows a path in the Scratch file system without permission to access it and tries to remove it. Instead of gracefully handling this issue, Bits&Bytes produces an exception and makes the whole computation fail. This is fixed by ignoring this kind of exception (`bitsandbytes/bitsandbytes/cuda_setup/main.py,` function `remove_non_existent_dirs`).

`Llama-recipes` is a repository containing ready-made code to execute Llama-2. Despite being relatively easy to use, some modifications have been applied to obtain the desired behavior.

Firstly, the saving functionalities of a distributed model (FSDP, Fully Sharded Data Parallelism) did not match the expected behavior. By modifying the checkpoint handling logic, this library now handles the saving of a distributed model equivalently as a sequential pre-trained model: the different shards are collected, converted into half-precision, and saved in the command-line specified folder in the same format as HuggingFace pre-trained models (`llama-recipes/src/llama_recipes/config/fsdp.py`, `checkpoint_type: StateDictType = StateDictType.FULL_STATE_DICT`, and `llama-recipes/src/llama_recipes/model_checkpointing/checkpoint_handler.py`, function `save_model_checkpoint`).

Secondly, the logic for loading the pre-processed, tokenised datasets has been implemented. This required to apply easily interpretable modifications to multiple files (`llama-recipes/src/llama_recipes/config/datasets.py`, `llama-recipes/src/llama_recipes/datasets/__init__.py`, `llama-recipes/src/llama_recipes/utils/dataset_utils.py`) and the creation of two new classes specifically designed for loading a specific amount of data (hard-coded in the classes themselves via `TRAIN_SAMPLES` and `VAL_SAMPLES`) from the provided tokenised datasets (`llama-recipes/src/llama_recipes/datasets/clean_mc4_it_dataset.py`, `llama-recipes/src/llama_recipes/datasets/mc4_cs_dataset.py`).

Thirdly, since the `--use_fast_kernels` converts the model structure into the BetterTransformer format, the logic for saving the sharded model broke again since the reverse conversion of the model to the Transformer format crashes. To this end, once the fast kernels are chosen, the model is copied before being converted so that when it comes the time to save the sharded model to disk, the old model can still be used as a reference for the original model structure. This quick fix required to propagate the copy of the original model through all the methods stack (`llama-recipes/src/llama_recipes/finetuning.py`, `llama-recipes/src/llama_recipes/utils/train_utils.py`, `llama-recipes/src/llama_recipes/model_checkpointing/checkpoint_handler.py`, reference the `model_old` variable).

Fourthly, the `llama-recipes/src/llama_recipes/finetuning.py` loads the tokeniser from the same folder as the Llama model. Since the saved checkpoints do not also contain the tokeniser model, all the tokeniser-related files contained in the original `llama-2-7b` folder have been moved to the `tokeniser` folder, and an additional parameter (`--tokeniser`) has been added to the script to load the tokeniser from the provided path (`llama-recipes/src/llama_recipes/finetuning.py`, `llama-recipes/src/llama_recipes/configs/training.py`).

Fifthly, different sharding strategies are required by different infrastructures. Due to the larger GPU RAM, Leonardo can bear the most advanced and efficient sharding technique available, `_HYBRID_SHARD_ZERO2`, while Karolina, having less GPU RAM, can bear the second most efficient sharding strategy, `HYBRID_SHARD`. Such values should be inserted in the FSDP configuration file (`llama-recipes/src/llama_recipes/configs/fsdp.py`) and are not specified a priori to avoid using the wrong sharding strategy. Furthermore, these sharding strategies reduce the inter-node communication overhead, effectively improving scaling performance over multiple nodes.

### AMD-specific modifications

A couple of quick fixes are needed to run Llama-2 on AMD-based infrastructures (e.g., Pawsey's Setonix). The practitioner should consider that such platforms use the ROCM software stack (instead of CUDA), and thus, some Python packages designed assuming a CUDA backend could require some work to be adapted to this different backend.

Notably, the 8-bit quantisation library Bits&Bytes still does not support the ROCm backend at the time of writing, requiring moving to an unofficial porting of Bits&Bytes supporting the ROCm environment: [Bits&Bytes-ROCm](https://git.ecker.tech/mrq/bitsandbytes-rocm). This software has been compiled (it requires the shader ISA code of the underlying AMD GPU, for example, gfx90a for the AMD Radeon Instinct MI250X, ROCm version 5.6, nvcc, and the HiPcc compiler) and substituted to the standard pip version of Bits&Bytes, effectively allowing the training process.

Furthermore, the `llama-recipes` code implicitly assumes an underlying NVIDIA/CUDA environment and, thus, explicitly checks for the installed CUDA version to decide if brain float support is available or not, thus implicitly disabling it on AMD/ROCm environments and forcing FP32 precision. To avoid this, the checks on the cuda version present in `llama-recipes` have been commented out (`llama-recipes/src/llama_recipes/utils/train_utils.py`, method `get_policies`), effectively enabling brain float 16bit mixed precision training also on AMD GPUs.

### Bypassing torchrun

Torchrun (also known as Elastic Launch) is a utility made available by PyTorch allowing to running multi-process PyTorch code without having the user worry about the actual distributed run details, such as the rank assignment, the process groups creation in case of multi-node multi-gpu training, and so on. Furthermore, it provides the so-called "elasticity", a soft form of fault tolerance, allowing the graceful handling of non-responding PyTorch processes, eventually restarting them. Unfortunately, this utility relies on a client-server architecture based on TCP communication, which can cause troubles in HPC environments where the number of TCP connections can be limited. Since our use case does not need fault-tolerance properties, we eliminated the torch run dependency by setting the needed environmental variable ourselves. We first investigated how torchrun sets them (`torchrun_env_var`) and then replicated the behavior through a static bash script, not relying on TCP communication (`launch.slurm`, which in turn calls `torchrun.sh`, can be found in the `script_leonardo` or `script_karolina` folder). This approch effectively allow us to scale over the 128 nodes limit we encouterend on Leonardo, exploiting the InfiniBand interconnection network.

### Minor fixes

HuggingFace `datasets` libraries connects to the network bu default. This fact can be annoing when working on HPC infrastructures, where nodes do not have internet connection. From a practical perspective, this behaviour can create problem when trying to access datasets. The `HF_DATASETS_OFFLINE` environment variable can be set to `1` to bypass internet requestes, thus not wasting time waiting for the timeovers. Note that, if working offline, the datasets should already been downloaded and available. If that is the case, loading a dataset simply requires to insert its local path in the `load_dataset` function in the `llama_recipes/datasets` dataset's file instead of the dataset name.

Some minor warning have been fixed to obtain a cleaner execution. In `transformers/utils/generic.py:441` and `transformers/utils/generic.py:441` the deprecated `torch.utils._pytree._register_pytree_node` has ben substituted with `torch.utils._pytree.register_pytree_node`; also, the deprecated `prepare_model_for_int8_training` in `llama_recipes/finetuning.py` has been substituted with `prepare_model_for_kbit_training`. In `llama_recipes/utils/memory_utils.py` the deprecated `torch.cuda.reset_max_memory_allocated` has been sustituted with `torch.cuda.reset_peak_memory_stats`.
