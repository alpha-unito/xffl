# xFFL Examples Overview

This repository provides **two categories of xFFL examples**, designed to demonstrate how xFFL can be used for federated and distributed training workflows in both cross-facility and intra-silo scenarios.

## 1. [Cross-Facility Federated Learning Example](cross-facility/)

This example targets **federated pre‑training of LLaMA or Mixtral models across multiple HPC facilities**.

It demonstrates how to:
- orchestrate **cross‑facility training** using
xFFL
- run compute tasks on heterogeneous HPC systems
- use SLURM, Singularity/Apptainer, and StreamFlow to coordinate the execution
- deploy massively distributed LLM training across federated compute sites

This is the most advanced example and reflects a real multi‑HPC federated setup.

## 2. [Intra‑Silo Distributed Training Examples (Three Levels)](intra-silo/)

Three increasingly complex intra‑silo (single‑facility) examples are provided to show how xFFL can be used to deploy distributed training workflows **within a single HPC or cluster**.

Each showcases a different model class and level of complexity:

### **a. MLP Example (Beginner Level)**

-   Minimal setup
-   Useful for understanding xFFL workflow basics
-   Ideal for testing infrastructure and debugging environments

### **b. CNN Example (Intermediate Level)**

-   Adds multi‑GPU training patterns
-   Demonstrates dataset handling and more realistic training behavior

### **c. LLM Example (Advanced Level)**

-   Demonstrates distributed LLM training inside one facility
-   Showcases large‑scale coordination and resource usage
-   Ideal for preparing deployments before scaling to cross‑facility mode

## Purpose of These Examples

Together, these examples illustrate:

-   How to scale from simple local distributed jobs to **full multi‑HPC federated LLM training**
-   How xFFL manages workflows at different levels of complexity
-   How to structure training code and argument parsing for compatibility with xFFL
-   How to prepare, test, and deploy distributed training on HPC infrastructures
