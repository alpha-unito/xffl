# xFFL Large Language Model (LLM) Pre‑Training Example

This example demonstrates how to pre‑train a **Large Language Model (LLaMA/Mixtral family)** using **xFFL** in an HPC environment. It showcases how to configure transformer models, load large datasets, initialize multi‑node distributed execution (FSDP/HSDP/FL), and run scalable training using xFFL.

## Overview

This example includes:

* A **configuration file** (`config.py`) defining the LLM architecture (LLaMA or Mixtral), dataset paths, training hyperparameters, and distributed options.

* A **training script** (`training.py`) implementing a full xFFL-compliant LLM training pipeline with:
  * deterministic setup
  * distributed initialization (FSDP/HSDP/FL)
  * activation checkpointing
  * dataset loading from disk
  * AdamW optimizer + cosine decay schedule
  * optional WandB logging
  * multi‑node training via xFFL

This example is designed for **large‑scale pre‑training**, not fine‑tuning (no LoRA/QLoRA, no parameter‑efficient methods). Models are trained fully in **bfloat16**.

## File Structure

```
llm_example/
│
├── config.py          # LLaMA/Mixtral + clean-mC4-it configuration
├── training.py        # Main distributed LLM training script
```

## `config.py`

The configuration defines all aspects of the example.

### **Model**

Two architectures are provided:

* **LLaMA 3.1 8B** (`llama`)
* **Mixtral 8×7B** (`mixtral`)

Available model fields:

* `model_type`: HF model class (LlamaForCausalLM or MixtralForCausalLM)
* `decoder_layers`: transformer layer classes for auto‑wrapping
* `wrapping_policy`: FSDP wrap policy for transformer layers
* `path`: base storage path for saved model weights

### **Dataset**

This example uses **clean_mc4_it**, an Italian‑language subset of mC4.

`cleanmc4it` specifies:

* dataset name
* train/val splits
* dataset base path

**Important:** dataset must already be tokenized and saved via `datasets.save_to_disk()`.

### **Training Configuration (`xffl_config`)**

Includes:

* logging level (same as ```python.logging```)
* random seed
* distributed strategy (`hsdp`, `federated_scaling`)
* attention backend (e.g., `sdpa`)
* subsampling for debugging
* batch sizes
* workers
* learning rate + AdamW betas
* optional LR scaling by world size
* output paths for checkpoints
* WandB configuration
* number of epochs (default: 1)
* federated batching (i.e., how many batches are processed between two FL aggregations)

This object is passed directly into `training.py`.

## `training.py`

This is a complete distributed LLM pre‑training script.

### **1. Distributed Setup**

Initializes multi‑node execution through:

```
distributed.setup_distributed_process_group(
    hsdp=config.hsdp,
    federated=config.federated_scaling,
    streams=config.cuda_streams,
)
```

Supports:

* Standard FSDP
* HSDP hierarchical groups
* Federated Learning scaling
* Multiple CUDA-streams usage to optimize GPU communications


### **2. Model Loading**

Models are loaded from disk:

```
AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    use_cache=False,
    local_files_only=not config.online,
    attn_implementation=config.attention,
    dtype=torch.bfloat16,
    device_map=state.init_device,
)
```

Weights can optionally be **re‑initialized** for true pre‑training.

### **3. FSDP/HSDP Wrapping + Activation Checkpointing**

```
model = modelling.create_fsdp_model(...)
utils.set_activation_checkpointing(model=model, layer=config.model.decoder_layers)
```

Ensures memory‑efficient training of multi‑billion‑parameter models.

### **4. Dataset Loading + Dataloaders**

Datasets are loaded via HF `load_from_disk`:

```
datasets = data.load_datasets_from_disk(...)
```

Distributed samplers ensure sharding and reproducibility.

### **5. Optimizer + Scheduler**

Uses **AdamW** with a simplified **LLaMA‑style cosine schedule** (warmup + cosine decay).

### **6. Training Loop**

Training is executed through xFFL’s:

```
processing.distributed_training(...)
```

This handles:

* training
* evaluation
* checkpointing
* federated batch aggregation
* WandB metric logging

### **7. Cleanup**

Distributed groups and objects are properly destroyed.


## Installation

You must also have **xFFL**, **PyTorch**, **Transformers**, and a properly configured HPC environment.

## Running the Example

After allocating nodes and activating your environment:

### Basic execution

```bash
xffl exec training.py config.py
```

### Debug mode

```bash
xffl --debug exec training.py config.py
```

xFFL will:

* initialize distributed execution
* preload large weights
* load the selected LLM
* wrap it with FSDP/HSDP/FL
* load the specified dataset
* start multi‑node training

## Dependencies

PyTorch, xFFL, and WandB must already be installed.


## Notes

* This is an **advanced example**: training LLaMA/Mixtral models requires significant GPU memory.
* Intended for **research HPC clusters**, not consumer hardware.
* All paths in the config file should be adapted to match local scratch/project directories.
* The script is designed for **pre‑training** but can be adapted for fine‑tuning.
