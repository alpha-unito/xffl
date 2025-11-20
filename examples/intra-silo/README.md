# xFFL Intra‑Silo Examples

This directory contains **three fully working xFFL examples** demonstrating how to execute **intra‑silo federated deployments** that seamlessly scale across any number of **HPC nodes** allocated via **SLURM**. These examples are designed to illustrate how xFFL can automatically distribute and manage training through **FSDP**, **HSDP**, and **FL** strategies.

The available examples are:

* **Simple MLP** – multilayer perceptron training on MNIST.
* **CNN** – a ResNet‑18 training on CIFAR‑10.
* **LLM** – a Llama‑3.1‑8B training on the *clean_mc4_it* dataset.

Each example demonstrates best practices for scaling and orchestrating training inside a single silo.

---

## Requirements

* Python **3.8+** (Python **3.11 recommended**)
* Access to an HPC cluster managed by **SLURM**
* A Python virtual environment *(recommended)* or a Docker/Singularity container
* SSH connectivity available among compute nodes

---

## Folder Structure

Each example folder contains:

* **training.py** – main training script that demonstrates how to use the xFFL API to build automatically scalable deployments.
* **config.py** – training configuration, including model, dataset, hyperparameters, and execution strategy.
* **requirements.txt** – additional Python dependencies required by the example (optional; extends the main xFFL `requirements.txt`).

---

## How to Run an Example

### 1. Set up a virtual environment with xFFL installed

Clone the repository:

```bash
git clone git@github.com:alpha-unito/xffl.git
cd xffl/
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install xFFL:

```bash
pip install .
```

---

### 2. Install example‑specific dependencies

Move into the target example folder and install optional dependencies:

```bash
cd examples/intra-silo/01_simple-MLP/
pip install -r requirements.txt
```

---

### 3. Acquire HPC nodes via SLURM

Before running an example, allocate one or more nodes:

```bash
srun --job-name=xFFL-example --nodes=2 --exclusive --pty bash
```

> ⚠️ **Important:** SLURM configurations vary across HPC systems. Adjust flags according to your cluster's requirements.

If accessing the nodes resets your environment, re‑activate the virtual environment containing xFFL and navigate back to the example folder.

---

### 4. Run the example

Launch training with:

```bash
xffl exec training.py config.py
```

This command deploys a distributed xFFL execution across all available nodes using the strategy defined in `config.py`.

You can enable debug output via:

```bash
xffl --debug exec training.py config.py
```

or:

```bash
xffl exec --debug training.py config.py
```

*TODO:* Add documentation for facility configuration.

---

## Notes

* These examples serve as a reference for intra‑silo xFFL usage.
* For production deployments, adapt scripts, cluster parameters, and resource allocations to suit your environment.
