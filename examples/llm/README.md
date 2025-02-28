# xFFL example - LLaMA training

This folder contains all the necessary files to run a LLaMA training, both _locally_ (distributed single-node, multi-GPU training) and _remotely_ (cross-facility federated learning).
The local simulation feature of xFFL is particularly useful to test a training script on a facility before launching it remotely.


## Prerequisites
Both deployments require a _model_, a _dataset_, and a _virtualization technology_ available on the deployment site. 
A _Singularity image_ (.sif) will work for both scenarios, while a _Python virtual environment_ is also acceptable for local simulations.
Both alternatives should come with the xFFL package already installed, together with any other Python package required by the training script.

A _command line argument parser_ based on `xffl.custom.parser` should be provided in a _separate source file_ from the main training script.
This is _mandatory_ for the remote execution of the training script and only suggested for local simulations.
The `xffl.custom.parser` is a standard Python `ArgumentParser`, with a few arguments mandatory for the cross-facility deployment already inserted by default.

A LLaMA model can be downloaded from the [official META LLaMA website](https://www.llama.com/llama-downloads/) or through [HuggingFace](https://huggingface.co/docs/transformers/model_doc/llama); this examples has been tested with LLaMA-3.1 8B.
This example has been tested on the tiny split of the [clean_mc4_it dataset](https://huggingface.co/datasets/gsarti/clean_mc4_it) by [gsarti](https://huggingface.co/gsarti), containing a cleaned version of Common Crawl's web crawl corpus (mC4). 


## Local simulation
Running a local LLaMA training through xFFL is simple and straightforward.
You just have to launch the training script (in this case, `client/src/training.py`) through the xFFL `simulate` feature; this will result in a local distributed training in which different training processes collaborate.
In this example, the distributed training is handled through [PyTorch's Fully Sharded Data Parallelism](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html).

As a first step, clone this repository on the computing infrastructure (this example uses SSH; if you do not have it configured, use HTTPS):
```bash
git clone git@github.com:alpha-unito/xffl.git
```

Create a local virtual environment and install xFFL:
```bash
cd xffl/
python -m venv .venv
source .venv/bin/activate
pip install .
```

Optionally, install flash attention (this example uses it by default, but the type of attention layer can be specified as a command line parameter; check the [requirements](https://pypi.org/project/flash-attn/))(not supported on ROCm at the time of writing):
```bash
pip install flash-attn --no-build-isolation
```

It is now time to run our training script through xFFL.
The `xffl` command offers a subcommand specifically designed to run single-node, multi-GPU training scripts: `simulate`.
`xffl simulate` accepts many command line arguments, but, in this example, only four will be used:
* The __mandatory__, positional argument: training script;
* `--world-size`: number of training processes to spawn, typically the number of GPUs available locally;
* `--venv`: path to the Python virtual environment _activate_ script;
* `--arguments`: list of command line arguments to be passed to the training script.

`training.py` requires at least two command line arguments to work correctly:
* `--model`: path to the LLaMA folder;
* `--dataset`: path to the already tokenized dataset folder;

Launching a single-node, multi-GPU simulation of a LLaMA training thus requires to run just this simple command:
```bash
xffl simulate examples/llama/client/src/training.py \
 --world-size _number_of_GPUs_ \
    --venv .venv/bin/activate \
 --arguments \
    --model /path/to/llama \
 --dataset /path/to/dataset
```

`xffl simulate` is particularly useful for testing and debugging a deployment on a target facility before running it remotely in a federated learning context.


## Remote deployment
It is time to run a distributed, [cross-facility federated learning deployment of LLaMA](https://hpc4ai.unito.it/hpc-federation/).
This example assumes that the target facilities are High-Performance Computers (HPCs) handled through the SLURM workload manager, while the launching facility is a cloud environment capable of accessing the HPCs through SSH.

### Prerequisites
Before starting the federated training, the tokenized training data on the training facilities, the model on the launching facility, and Singularity images on both sites must already be available.
Simple Singularity image definitions are given for both the aggregator (launching facility) and the client (training facility).

#### Project creation
`xffl` offers the `config` subcommand, specifically designed to guide the user in creating a valid xFFL project.
To start this process, just run the following command on the launching facility:
```bash
xffl config
```
Follow and answer the prompts you are asked; this process will create a project folder containing everything needed for xFFL to run a cross-facility federated learning smoothly.

#### Project launching
It is now time to run the created project on the target facilities:
```bash
xffl run -p _project_folder_
```
This will launch the StreamFlow WMS and deploy the specified training on the selected facilities.
