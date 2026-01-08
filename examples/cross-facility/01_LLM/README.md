# xFFL – LLaMA Cross-Facility & Local Training Example

This directory contains all the components required to train a **LLaMA model** using **xFFL**, both:

- **Locally** — single-node, multi-GPU *distributed training*, ideal for debugging and validation.
- **Remotely** — *cross-facility federated learning*, orchestrating multiple heterogeneous HPC systems.

The **local simulation** feature is especially useful for testing and validating your training pipeline **before** launching a full remote deployment.

## Prerequisites

Both local and remote deployments require:

### **Model**
A LLaMA model directory.
You may download it from:

- Meta’s official website: https://www.llama.com/llama-downloads/
- HuggingFace: https://huggingface.co/docs/transformers/model_doc/llama

This example has been tested with **LLaMA-3.1 8B**.

### **Dataset**
A tokenized dataset compatible with the training script.
This example uses the *tiny* split of:

**clean_mc4_it**
https://huggingface.co/datasets/gsarti/clean_mc4_it

### **Runtime Environment**
Each site must provide:

- A **Python virtual environment** (for local simulations)
  **or**
- A **Singularity/Apptainer container** (.sif) with:
  - xFFL installed
  - all project-specific libraries installed

### **Custom Argument Parser**

The training script must receive its arguments through a **stand-alone parser module** built using:

```
xffl.custom.parser
```

This is:

- **mandatory** for cross-facility deployments
- **recommended** for local simulations

The parser extends a standard Python `ArgumentParser` with additional arguments required by xFFL.

## Local Simulation (single-node, multi-GPU)

Local simulation is the easiest way to validate your LLaMA training script.
It uses the `xffl simulate` command to spawn multiple training processes that collaboratively train the model using **PyTorch FSDP**.

### 1. Clone the repository

```bash
git clone git@github.com:alpha-unito/xffl.git
cd xffl/
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

### 3. (Optional) Install FlashAttention

> ⚠️ FlashAttention is **not supported on ROCm** at the time of writing.

```bash
pip install flash-attn --no-build-isolation
```

### 4. Run the training using `xffl simulate`

The command requires:

| Argument | Meaning |
|---------|---------|
| **training script** | Mandatory positional argument |
| `--world-size` | Number of spawned processes (usually = number of GPUs) |
| `--venv` | Path to the `activate` script of the Python environment |
| `--arguments` | Arguments forwarded to the training script |

The training script (`training.py`) expects at least:

```
--model    /path/to/llama
--dataset  /path/to/tokenized_dataset
```

### Example command

```bash
xffl simulate examples/llama/client/src/training.py     --world-size <NUM_GPUS>     --venv .venv/bin/activate     --arguments         --model /path/to/llama         --dataset /path/to/dataset
```

Local simulation is ideal to debug:

- dataset loading
- distributed initialization
- FSDP wrapping
- memory usage
- training correctness

before deploying across multiple facilities.

## 🌍 Remote Cross-Facility Deployment

This section describes how to run **federated LLaMA training across multiple HPC clusters**, using:

- **Singularity** (clients + aggregator)
- **SLURM** (client execution)
- **StreamFlow** (workflow orchestration)
- **SSH** access (from aggregator to each client facility)

This example mirrors real multi-facility deployments such as:
https://hpc4ai.unito.it/hpc-federation/

---

## Remote Requirements

Before launching the federated deployment, ensure:

### On the **training facilities** (clients)
- Tokenized dataset is available
- `client.sif` Singularity image is available
- SLURM access works

### On the **launching/aggregator facility**
- LLaMA model is available
- `aggregator.sif` image is available
- SSH access to all client HPCs is configured
- xFFL + StreamFlow are installed locally

Both aggregator and client Singularity recipes are provided in:

```
client/container/
aggregator/container/
```

## Project Creation

xFFL provides an interactive wizard to generate a valid multi-facility project directory.

Run:

```bash
xffl config
```

Follow the prompts to define:

- facilities
- SLURM settings
- container locations
- dataset/model paths
- authentication methods

A fully-configured **project folder** is then created automatically.

## Launching the Distributed Training

Once the project folder is ready, start the cross-facility workflow with:

```bash
xffl run -p <project_folder>
```

xFFL will:

1. Start StreamFlow
2. Connect to all client HPCs
3. Submit SLURM jobs
4. Exchange model parameters across facilities
5. Synchronize training rounds
6. Manage failures and retries

## Final Notes

- This example serves as a **reference implementation** for LLaMA training using xFFL in both local and cross-facility modes.
- Model and dataset paths must be configured manually based on your environment.
- Singularity definitions can be extended with additional libraries depending on your needs.
- Cross-facility performance depends heavily on network connectivity, remote queue times, and facility load.
