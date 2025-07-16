"""Command line argument parser for the xFFL-LLM example"""

import argparse
import logging

### Argument parser
parser = argparse.ArgumentParser(
    prog="Cross-Facility Federated Learning (xFFL) - simple MLP example",
    description="This xFFL example trains an MLP on multiple HPC compute nodes.",
)

parser.add_argument(
    "-s",
    "--seed",
    help="Random execution seed (for reproducibility purposes)",
    type=int,
    default=None,
)

parser.add_argument(
    "-e",
    "--epochs",
    help="Number of training epochs",
    type=int,
    default=1,
)

parser.add_argument(
    "-dbg",
    "--debug",
    help="Print of debugging statements",
    action="store_const",
    dest="loglevel",
    const=logging.DEBUG,
    default=logging.INFO,
)

parser.add_argument("-wb", "--wandb", help="Enable WandB", action="store_true")

parser.add_argument(
    "-name",
    "--wandb-name",
    help="WandB group name",
    type=str,
    default="Simple-MLP",
)

parser.add_argument(
    "-mode",
    "--wandb-mode",
    help="WandB mode",
    type=str,
    default="disabled",
    choices=["online", "offline", "disabled"],
)

parser.add_argument(
    "-t",
    "--train-batch-size",
    help="Training batch size",
    type=int,
    default=1024,
)

parser.add_argument(
    "-v",
    "--val-batch-size",
    help="Validation batch size",
    type=int,
    default=1,
)

parser.add_argument(
    "-ws",
    "--workers",
    help="Number of data loaders workers",
    type=int,
    default=0,
)

parser.add_argument(
    "-lr",
    "--learning-rate",
    help="Learning rate",
    type=float,
    default=1e-2,
)

parser.add_argument(
    "-om",
    "--output-model",
    help="Saved model name",
    type=str,
    default=None,
)

parser.add_argument(
    "-fs",
    "--federated-scaling",
    help="Enable Federated scaling with the specified federated group size(s)",
    type=int,
    nargs="+",
    default=None,
)

parser.add_argument(
    "-fb",
    "--federated-batches",
    help="Number of training batches to process between two federated averaging",
    type=int,
    default=1,
)

parser.add_argument(
    "-oc",
    "--one_class",
    help="Give only one MNIST class to each client",
    action="store_true",
)

parser.add_argument(
    "-hsdp",
    "--hsdp",
    help="Enable Hybrid Sharding Data Parallel (HSDP) with the specified replica group size",
    type=int,
    default=None,
)

parser.add_argument(
    "-c",
    "--cuda-streams",
    help=" Number of CUDA streams to instantiate",
    type=int,
    default=4,
)
