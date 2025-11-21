# Simple MLP xFFL Example

This example demonstrates how to train a **simple MLP model on MNIST** using **xFFL** within an HPC environment. It illustrates how to define a model and dataset configuration, initialize distributed execution, wrap the model with **FSDP/HSDP/FL**, and perform scalable training across one or more compute nodes.

## Overview

This example includes:

* A **configuration file** (`config.py`) defining the model architecture, dataset settings, and training hyperparameters.
* A **training script** (`training.py`) implementing a complete xFFL‑compatible workflow, including deterministic setup, distributed initialization, model wrapping, dataloaders, optimizer creation, and training.
* A minimal **requirements.txt** listing additional Python dependencies specific to this example.

The model used here is a small fully connected MLP, and the dataset is MNIST.

## File Structure

```
example/
│
├── config.py          # MLP + MNIST xFFL configuration
├── training.py        # Main training script with FSDP/HSDP support
└── requirements.txt   # Additional Python dependencies for the example
```

## `config.py`

The configuration file defines all components required for running the example.

### **Model**

A small fully connected MLP with the following layers:

* `fc1`: 784 → 64
* `fc2`: 64 → 32
* `fc3`: 32 → 10

The model is wrapped using the `ModelInfo` xFFL dataclass, ensuring compatibility with the framework.

### **Dataset**

MNIST is loaded through a custom `DatasetInfo` subclass. The dataset path must be provided by the user.

### **Training Configuration** (`xffl_config`)

This dataclass aggregates all high‑level training options, including:

* logging level (following `python.logging` semantics)
* random seed for deterministic execution
* distributed strategy (FSDP, HSDP, or FL)
* train/validation batch sizes
* learning rate
* number of epochs (default: 10)
* number of dataloader workers
* optional one‑class training and dataset subsampling

By default, FSDP is used across all available nodes. If `hsdp=n` or `federated_scaling=n` is provided, xFFL will configure HSDP or FL execution with node groups of size `n`.

Users are encouraged to customize the configuration according to their experiments.

## `training.py`

The core training script performs the full distributed workflow.

### **Distributed Setup**

Initializes PyTorch’s distributed process group via:

```
xffl.distributed.setup_distributed_process_group(
    hsdp=config.hsdp,
    federated=config.federated_scaling
)
```

This establishes the distributed state needed for FSDP, HSDP, or FL execution.

### **Model Initialization + FSDP Wrapping**

The model is instantiated from the configuration and wrapped using:

```
xffl.modelling.create_fsdp_model(model, state)
```

Each rank receives its GPU assignment automatically.

### **Dataset + Dataloaders**

MNIST is downloaded (if needed), preprocessed, optionally filtered (one‑class mode), optionally subsampled, and loaded into distributed‑enabled dataloaders.

### **Training Loop**

The model is trained using:

* `Adadelta` optimizer
* `NLLLoss` criterion
* xFFL’s `processing.distributed_training` utilities

Training and validation metrics are logged throughout.

### **Cleanup**

At the end of the run, the script cleans up the distributed group and releases GPU/CPU resources.

## Installation

Inside the example directory:

```bash
pip install -r requirements.txt
```

You must also have **xFFL installed** in your active Python environment.

## Running the Example

After allocating HPC nodes and activating your environment, run:

```bash
xffl exec training.py config.py
```

xFFL will:

* initialize the distributed environment
* wrap the MLP model using FSDP/HSDP/FL
* run distributed MNIST training
* aggregate results across ranks

Enable debug logging with:

```bash
xffl --debug exec training.py config.py
```

## Dependencies

The example requires only:

```
torchvision
```

PyTorch and xFFL must already be installed.

## Notes

* This is a **minimal template** showing how to integrate a simple model and dataset into xFFL.
* All components—model, dataset, and training parameters—are configurable in `config.py`.
* The example supports **CPU and GPU**, scaling automatically across multiple nodes when launched via SLURM.
