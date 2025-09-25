"""Command line argument parser for the xFFL-LLM example"""

import logging

from xffl.custom.parser import ArgumentParser

### Argument parser
parser = ArgumentParser(
    prog="Cross-Facility Federated Learning (xFFL) - LLM example",
    description="This xFFL example pre-trains an LLM on multiple HPC infrastructures.",
)

parser.add_argument(
    "-attn",
    "--attention",
    help="Type of attention implementation to use",
    type=str,
    default="sdpa",
    choices=["sdpa", "eager", "flash_attention_2"],
)

parser.add_argument("-on", "--online", help="Online mode", action="store_true")

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
    default="LLaMA-3.1 8B",
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
    "-sub",
    "--subsampling",
    help="Quantity of data samples to load (for each dataset)",
    type=int,
    default=0,
)

parser.add_argument(
    "-t",
    "--train-batch-size",
    help="Training batch size",
    type=int,
    default=2,
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
    default=2,
)

parser.add_argument(
    "-lr",
    "--learning-rate",
    help="Learning rate",
    type=float,
    default=3e-4,
)

parser.add_argument(
    "-wd",
    "--weight-decay",
    help="Weight decay",
    type=float,
    default=0.1,
)

parser.add_argument(
    "-sz",
    "--step-size",
    help="Learning rate scheduler step size",
    type=int,
    default=1,
)

parser.add_argument(
    "-g",
    "--gamma",
    help="Learning rate scheduler gamma",
    type=float,
    default=0.85,
)

parser.add_argument(
    "-om",
    "--output-model",
    help="Saved model name",
    type=str,
    default=None,
)

parser.add_argument(
    "-hsdp",
    "--hsdp",
    help="Enable Hybrid Sharding Data Parallel (HSDP) with the specified replica group size",
    type=int,
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
    "-b",
    "--benchmark",
    help="Benchmark the aggregation strategies",
    type=int,
    default=None,
)

parser.add_argument(
    "-c",
    "--cuda-streams",
    help="Number of CUDA streams to instantiate",
    type=int,
    default=4,
)

parser.add_argument(
    "-csv",
    "--csv",
    help="Dump the benchmark results to the provided csv file",
    type=str,
    default="output.csv",
)

parser.add_argument(
    "-slr",
    "--scale-learning-rate",
    help="Scale the learning rate in the numbers of models replicas",
    action="store_true",
)
