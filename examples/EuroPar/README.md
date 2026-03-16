# xFFL — EuroPar 2026 — FL + *SDP Experiments

This repository contains the source code required to reproduce the large-scale pre-training experiments presented at **EuroPar 2026**, based on **federated-learning-augmented sharded data parallelism** (FL + *SDP).

The experiments train **Llama 3.1-8B** on the **clean_mc4_it** dataset using the xFFL framework on a large HPC system.

> 🧪 Original experiments were executed on **128 Leonardo HPC nodes (512 NVIDIA A100 GPUs)**.

## Overview

This repository provides everything needed to reproduce the training runs evaluated in the paper.

### Included components

* **Four configuration files**, one for each parallelization strategy:

  * `config_FSDP.py` — Fully Sharded Data Parallel (FSDP)
  * `config_HSDP.py` — Hierarchical Sharded Data Parallel (HSDP)
  * `config_FL+FSDP.py` — Federated Learning + FSDP
  * `config_FL+HSDP.py` — Federated Learning + HSDP

  Each configuration defines:

  * model parameters
  * dataset paths
  * optimizer settings
  * distributed training options
  * logging configuration

* **Training script (`training.py`)**

  Implements a full xFFL-compliant large-scale LLM pre-training pipeline with:

  * deterministic initialization for reproducibility
  * distributed setup (FSDP / HSDP / FL+FSDP / FL+HSDP)
  * dataset loading and preprocessing
  * optional Weights & Biases logging
  * multi-node orchestration via xFFL

## Training Setup

* **Model:** Llama 3.1-8B
* **Dataset:** clean_mc4_it
* **Precision:** bfloat16
* **Training type:** full pre-training (not fine-tuning)

⚠️ Parameter-efficient methods such as LoRA or QLoRA are **not used**.

## Repository Structure

```
EuroPar/
│
├── config_FSDP.py          # LLaMA 3.1-8B + clean_mc4_it using FSDP
├── config_HSDP.py          # LLaMA 3.1-8B + clean_mc4_it using HSDP
├── config_FL+FSDP.py       # LLaMA 3.1-8B + clean_mc4_it using FL+FSDP
├── config_FL+HSDP.py       # LLaMA 3.1-8B + clean_mc4_it using FL+HSDP
├── training.py             # Main distributed training script
```

## Requirements

### Software

* Python environment compatible with xFFL
* xFFL installed and properly configured
* PyTorch with distributed support
* Access to the Llama 3.1-8B weights
* Access to the clean_mc4_it dataset

### Hardware

This code targets large HPC systems.

At minimum:

* Multi-node GPU cluster
* High-speed interconnect (e.g., InfiniBand)
* SLURM or compatible scheduler
* Sufficient storage bandwidth for dataset streaming

## Installation

Install and configure xFFL according to its official documentation.

Ensure that:

* distributed communication works across nodes
* filesystem paths are accessible from all nodes
* required datasets and model checkpoints are available

## Running the Experiments

After allocating compute nodes and activating your environment:

### Basic execution example

```bash
xffl exec training.py config_FSDP.py
```

### What happens during execution

xFFL will automatically:

* detect the allocated resources (e.g., via SLURM)
* initialize distributed training
* load the Llama 3.1-8B model from the configured path
* wrap the model with the selected parallelization strategy
* load the dataset
* start synchronized multi-node training

## Configuration

All experiment-specific parameters are defined in the configuration files.

Before running:

* Update dataset paths
* Update model checkpoint paths
* Adjust output/logging directories
* Verify cluster-specific settings

⚠️ Paths must be valid on all participating nodes.


## Reproducibility Notes

* Deterministic initialization is enabled where possible
* Performance and scalability results depend on hardware topology
* Exact reproduction requires a system comparable to the Leonardo cluster

## Limitations

* Designed for **pre-training workloads**, not fine-tuning
* Assumes availability of large GPU clusters
* Not optimized for single-node execution

## Adapting the Code

Although tailored for pre-training, the pipeline can be extended to support:

* fine-tuning
* alternative datasets
* different model sizes
* custom distributed strategies
