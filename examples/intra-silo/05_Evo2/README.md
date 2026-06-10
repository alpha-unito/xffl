# xFFL Evo2 Genomics Pre‑Training Example

This example demonstrates how to train the **Evo2 1B Base** genomic foundation model using **xFFL** on an HPC cluster. The configuration is tailored for distributed training with FSDP and focuses on genomic sequence modeling using the OpenGenome dataset and a character-level tokenizer.

## Overview

The example contains:

- `config.py` – complete xFFL configuration, including model loading, dataset definition, optimizer, scheduler, and experiment settings.
- `training.py` – xFFL-compatible training script (expected to be executed through `xffl exec`).

Key characteristics:

- Evo2 1B Base model (`evo2_1b_base`)
- Genomic sequence modeling
- Character-level tokenization
- bfloat16 training
- AdamW optimizer
- Warmup + cosine learning-rate schedule
- Offline Hugging Face execution
- Distributed HPC execution through xFFL

## File Structure

```text
project/
├── config.py
├── training.py
├── requirements.txt
└── README.md
```

## Model Configuration

### Evo2 1B Base

The configuration defines the model through the `evo_2` dataclass.

Main settings:

| Parameter | Value |
|-----------|--------|
| Model | Evo2 1B Base |
| Internal name | `evo2_1b_base` |
| Attention backend | `flash_attention_2` |
| Precision | bfloat16 |
| Activation checkpointing | Disabled |
| Weight source | Local checkpoint |
| Hugging Face mode | Offline |

The model is loaded from the Evo2 configuration file contained in the evo2.utils package:

```python
configs/evo2-1b-8k.yml
```

and instantiated through the StripedHyena implementation.

## Dataset Configuration

### OpenGenome

Dataset identifier:

```text
opengenome2-metagenomes-plantcad2-c4096
```

Dataset splits:

- Train
- Validation

Data are loaded from local parquet files.

### Tokenization

The dataset uses:

```python
CharLevelTokenizer(512)
```

Training samples are transformed into:

- Inputs: all tokens except the last
- Targets: all tokens except the first

This enables standard next-token prediction training.

### Dataset Parameters

| Parameter | Value |
|-----------|--------|
| Train batch size | 2 |
| Validation batch size | 1 |
| Train subsampling | 1000 samples |
| Validation subsampling | 20 samples |
| Workers | 2 |

## Training Configuration

### General

| Parameter | Value |
|-----------|--------|
| Seed | 42 |
| Epochs | 1 |
| Gradient accumulation | 1 |
| Gradient clipping | 1.0 |

### Optimization

Optimizer:

```python
AdamW
```

Parameters:

| Parameter | Value |
|-----------|--------|
| Learning rate | 1e-6 |
| Weight decay | 0.1 |
| Betas | (0.9, 0.95) |
| Fused optimizer | Enabled |

### Loss Function

Cross-entropy loss with:

```python
ignore_index = 0
```

applied to next-token prediction.

## Learning Rate Scheduler

The example implements a custom scheduler featuring:

1. Linear warmup
2. Cosine decay
3. Gradient-accumulation awareness

Scheduler parameters:

| Parameter | Value |
|-----------|--------|
| Warmup fraction | 1% |
| Final LR ratio | 0.1 (scheduler default) |

The configuration also exposes:

```python
final_lr_ratio = 0.01
```

for future customization.

## Distributed Training

The configuration is designed for xFFL distributed execution.

Features include:

- FSDP-compatible model loading
- Transformer auto-wrap policy for Evo2 attention blocks
- Mixed precision (bfloat16)
- Multi-node HPC execution

## Experiment Tracking

Weights & Biases settings:

| Parameter | Value |
|-----------|--------|
| Entity | alpha-unito |
| Project | xFFL playground |
| Group | 02_CNN |
| Run name | Example |
| Mode | disabled |

By default, WandB logging is disabled.

## Installation

Minimal dependency specified by the example:

```bash
pip install -r requirements.txt
```

Current requirements:

```text
evo2
```

A working xFFL installation, PyTorch distributed environment, and Evo2 dependencies are also required.

## Running

Execute the training through xFFL:

```bash
xffl exec training.py config.py
```

Debug mode:

```bash
xffl --debug exec training.py config.py
```

## Paths

The configuration assumes the following base path:

```text
/leonardo_scratch/fast/uToID_bench/xffl
```

Expected directory layout:

```text
BASE_PATH/
├── models/
│   └── evo2_1b_base/
└── data/
    └── opengenome2-metagenomes-plantcad2-c4096/
```

Adapt these paths to your local HPC environment before execution.

## Notes

- The example is intended for HPC systems.
- All training is performed in bfloat16.
- Hugging Face online access is disabled (`HF_HUB_OFFLINE=1`).
- The dataset and model checkpoints must already be available locally.
