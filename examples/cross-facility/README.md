# xFFL – Cross-Facility LLM Example

This directory contains a fully functional **cross-facility federated learning (FL) example** built with **xFFL**, designed to train a Large Language Model (LLM) across **multiple heterogeneous HPC facilities**.

It demonstrates how xFFL leverages:

* **Containerization** via Singularity/Apptainer
* **SLURM** for job scheduling
* **StreamFlow** for orchestrating cross-facility workflows

This example has been successfully tested on all **EuroHPC** systems:
**Deucalion, Discoverer, Karolina, Leonardo, LUMI, MareNostrum5, MeluXina, and Vega.**

## Folder Structure

```
llm/
│
├── aggregator/
│   ├── container/
│   │   └── aggregator.def     # Singularity recipe for the aggregator
│   └── src/                   # Aggregator-side Python source code
│
└── client/
    ├── container/
    │   └── client.def         # Singularity recipe for the clients
    ├── slurm_templates/       # SLURM submission templates for EuroHPC systems
    └── src/                   # Client-side Python source code
```

### Aggregator

The aggregator is responsible for:

* Coordinating federated learning rounds
* Receiving and aggregating client model updates
* Managing communication across HPC facilities
* Orchestrating workflow execution via StreamFlow

> The aggregator runs inside a Singularity/Apptainer container to ensure portability across different HPC systems.

### Client

Each client:

* Performs local training on a specific HPC cluster
* Loads dataset, model, and training parameters
* Trains for a configurable number of local steps/epochs
* Sends model updates back to the aggregator
* Is launched via SLURM using the provided templates

## ⚙️ Requirements

Before running this example, you need:

* Access to **two or more HPC facilities**
* Working **SLURM accounts** on each cluster
* **Singularity/Apptainer** installed on every HPC system
* SSH connectivity from the aggregator node to all client clusters
* Python environment with **xFFL** installed (on aggregator)
* StreamFlow configured for remote job submission (recommended for multi-facility orchestration)

## Building the Containers

### Aggregator Container

```bash
cd aggregator/container
sudo singularity build aggregator.sif aggregator.def
```

### Client Container

```bash
cd client/container
sudo singularity build client.sif client.def
```

> After building, transfer the `*.sif` images to their respective HPC clusters.

## SLURM Templates

The directory `client/slurm_templates/` contains ready-to-use SLURM scripts for the tested EuroHPC systems:

* deucalion
* discoverer
* karolina
* leonardo
* lumi
* marenostrum5
* meluxina
* vega

Each template includes:

* Resource requests (nodes, GPUs, CPUs, memory)
* Module/environment setup
* Container execution commands
* Cluster-specific job parameters

> Edit only if your allocation or environment differs from the defaults.

## Federated Learning Workflow

1. Aggregator broadcasts the **global model** to all clients
2. Clients perform **local training** on their assigned HPC resources
3. Clients send **model updates** back to the aggregator
4. Aggregator **aggregates updates** into a new global model
5. Repeat for multiple **federated rounds**

> This workflow enables true multi-HPC federated training of an LLM with containerized consistency.

## Notes & Tips

* This example serves as a **reference implementation** for cross-facility training with xFFL.
* Model architecture, dataset handling, and training specifics are defined in `src/` for both aggregator and clients.
* Modify the Singularity recipes to include additional libraries or dependencies if needed.
* Real-world performance depends on network throughput, inter-facility connectivity, and SLURM queue times.
* Use debug flags to monitor job execution across clusters:

```bash
xffl --debug exec training.py config.py
```
