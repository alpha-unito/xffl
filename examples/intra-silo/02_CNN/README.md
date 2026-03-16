# CNN xFFL Example

This example demonstrates how to train a **CNN** using **xFFL** in an HPC environment. It shows how to define a model and dataset configuration, initialize distributed execution, wrap the model with **FSDP/HSDP/FL**, and perform scalable training across one or more compute nodes.

## Overview

This example includes:

* A **configuration file** (`config.py`) defining the CNN model class, CIFAR‑10 dataset settings, and training hyperparameters.
* A **training script** (`training.py`) implementing a full xFFL‑compatible workflow: deterministic setup, distributed initialization, model wrapping, dataloaders, optimizer creation, training, and logging.
* A minimal **requirements.txt** specifying the additional dependencies (`torchvision`, `wandb`).

The model used here is **ResNet18**, and the dataset is **CIFAR‑10**.

## File Structure

```
example/
│
├── config.py          # ResNet18 + CIFAR‑10 xFFL configuration
├── training.py        # Main training script with FSDP/HSDP/FL
└── requirements.txt   # Additional Python dependencies
```

## `config.py`

The configuration file defines the components needed to run the example.

### **Model**

`Cnn` defines a ResNet18 architecture using:

* `class_ = models.resnet18`

It is wrapped as an xFFL‑compatible `ModelInfo` dataclass.

### **Dataset**

`Cifar` specifies the CIFAR‑10 dataset using an xFFL `DatasetInfo` subclass:

* dataset name: `CIFAR10`
* dataset path: `data/` (customizable by the user)

### **Training Configuration (`xffl_config`)**

This dataclass stores all high‑level training settings, including:

* logging level (`logging.DEBUG` by default)
* random seed for deterministic execution
* distributed strategy (FSDP by default; HSDP/FL via `hsdp` or `federated_scaling`)
* batch sizes (train: 1024, val: 1)
* learning rate (1e‑2) and momentum (0.9)
* number of epochs (10)
* number of dataloader workers (0)
* optional one‑class training
* optional dataset subsampling (commented out by default)
* full Weights & Biases configuration (disabled by default)

If `hsdp=n` or `federated_scaling=n` is provided, xFFL configures HSDP or FL groups of size `n`. Otherwise, standard FSDP across all nodes is used.

## `training.py`

The core script implementing the distributed training pipeline.

### **Distributed Setup**

The script initializes the PyTorch distributed backend via:

```
xffl.distributed.setup_distributed_process_group(
    hsdp=config.hsdp,
    federated=config.federated_scaling,
)
```

This prepares the distributed environment for FSDP, HSDP, or FL.

### **Model Initialization + FSDP Wrapping**

```
xffl.modelling.create_fsdp_model(model, state)
```

Each rank automatically receives the appropriate GPU assignment.

### **Dataset + Dataloaders**

CIFAR‑10 is downloaded, normalized, and optionally:

* filtered to a single class (`one_class` mode)
* subsampled (if `subsampling` is enabled)

Distributed (or standard) dataloaders are created for training and testing.

### **Training Loop**

The training is performed using:

* **SGD** optimizer (with configurable lr + momentum)
* **CrossEntropyLoss**
* xFFL’s `processing.distributed_training` utilities

Training/validation metrics are logged and optionally pushed to **Weights & Biases**.

### **Cleanup**

At the end, the script:

* finalizes W&B
* cleans up the distributed process group
* releases model/optimizer resources

## Installation

Inside the example directory:

```bash
pip install -r requirements.txt
```

You must also have **xFFL installed** in your environment.

## Running the Example

Once HPC nodes are allocated and your environment is active, run:

```bash
xffl exec training.py config.py
```

xFFL will:

* initialize the distributed backend
* wrap the ResNet18 model with FSDP/HSDP/FL
* launch distributed CIFAR‑10 training
* aggregate results across ranks

Enable debug logs with:

```bash
xffl --debug exec training.py config.py
```

## Dependencies

Additional requirements specific to this example:

```
torchvision
wandb
```

PyTorch and xFFL must already be installed.

## Notes

* This example serves as a **compact reference** for integrating CNN models and CIFAR‑10 into xFFL.
* All model, dataset, and training settings can be customized via `config.py`.
* Works on **CPU or GPU**, scaling seamlessly across nodes when launched through SLURM.
