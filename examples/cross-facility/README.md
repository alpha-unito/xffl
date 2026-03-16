# xFFL – Cross-Facility LLM Example

This directory contains a **cross-facility federated learning (FL) example** built with **xFFL**, designed to train a Large Language Model (LLM) across **multiple heterogeneous HPC facilities**.

It demonstrates how xFFL leverages:

* **Containerization** via Singularity/Apptainer
* **SLURM** for job scheduling
* **StreamFlow** for orchestrating cross-facility workflows

This example has been successfully tested on all **EuroHPC** systems:
**Deucalion, Discoverer, Karolina, Leonardo, LUMI, MareNostrum5, MeluXina, and Vega.**

## Example structure

```
01_LLM/
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

Usually, the aggregator is run on the same machine used to launch the entire xFFL deployment (usually a cloud VM). In this case, the aggregator infrastructure also orchestrates the workflow execution via StreamFlow.

### Client

Each client:

* Performs local training on a specific HPC cluster
* Loads dataset, model, and training parameters
* Trains for a configurable number of local steps/epochs
* Sends model updates back to the aggregator
* Is launched via SLURM using the provided templates

> The clients runs inside a Singularity/Apptainer container to ensure portability across different HPC systems.

## Requirements

Before running this example, you need:

* Access to **one or more HPC facilities**
* Working **SLURM accounts** on each cluster
* **Singularity/Apptainer** installed on every HPC system
* SSH connectivity from the aggregator node to all client clusters
* Working **xFFL** installation (on aggregator)

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

> It's not necessary to manually move the images to the clusters, StreamFlow will take care of it

## SLURM Templates

The directory `client/slurm_templates/` contains ready-to-use SLURM scripts for the tested EuroHPC systems:

* Deucalion
* Discoverer
* Karolina
* Leonardo
* LUMI
* MareNostrum5
* MeluXina
* Vega

Each template includes:

* Resource requests (nodes, GPUs, CPUs, memory)
* Module/environment setup
* Container execution commands
* Cluster-specific job parameters

> Edit these files to suite your account and needs.

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
