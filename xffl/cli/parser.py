"""Argument parser for xFFL

The basic parser offers informative functions.
All the advanced features are offered by the subcommands and the relative subparsers
"""

import argparse
import logging
import os

from xffl.custom.types import PathLike

# Base parser
parser = argparse.ArgumentParser(
    prog="xffl",
    description="Cross-Facility Federated Learning (xFFL) is a federated learning (FL) framework based on the StreamFlow workflow management system (WMS) developed in the Parallel Computing [Alpha] research group at the University of Turin, Italy.",
    add_help=False,
)

parser.add_argument(
    "-h", "--help", help="Show this help message and exit", action="store_true"
)

parser.add_argument("-v", "--version", help="Display xFFL version", action="store_true")

parser.add_argument(
    "-dbg",
    "--debug",
    help="Print of debugging statements",
    action="store_const",
    dest="loglevel",
    const=logging.DEBUG,
    default=logging.INFO,
)

### Subparsers ###
subparsers = parser.add_subparsers(dest="command", help="Available xFFL subcommands")

# Subcommand: config
config_parser = subparsers.add_parser(
    name="config",
    description="Guided xFFL experiment configuration",
    help="Guided xFFL experiment configuration",
    add_help=False,
)

config_parser.add_argument(
    "-h", "--help", help="Show this help message and exit", action="store_true"
)

config_parser.add_argument(
    "-p", "--project", help="Project name", type=str, default="project"
)

config_parser.add_argument(
    "-w",
    "--workdir",
    help="Working directory path",
    type=PathLike,
    default=os.getcwd(),
)

# Subcommand: run
run_parser = subparsers.add_parser(
    name="run",
    description="Run an xFFL experiment",
    help="Run an xFFL experiment",
    add_help=False,
)

run_parser.add_argument(
    "-h", "--help", help="Show this help message and exit", action="store_true"
)

run_parser.add_argument(
    "-w",
    "--workdir",
    help="Working directory path",
    type=PathLike,
    default=os.getcwd(),
)

run_parser.add_argument(
    "-p",
    "--project",
    help="Name of the project/Python execultable file",
    type=str,
    default="project",
)

run_parser.add_argument(
    "-o",
    "--outdir",
    help="Output directory",
    type=str,
    default=None,
)

run_parser.add_argument(
    "--quiet", help="Only prints results, warnings and errors", action="store_true"
)

run_parser.add_argument(
    "-args",
    "--arguments",
    help="Command line arguments to be passed to the executable",
    type=str,
    nargs="+",
    default=[],
)


# Subcommand: simulate
simulate_parser = subparsers.add_parser(
    name="simulate",
    description="Run a script locally through xFFL",
    help="Run a script through xFFL",
    add_help=False,
)

simulate_parser.add_argument(
    "-h", "--help", help="Show this help message and exit", action="store_true"
)

simulate_parser.add_argument(
    "executable",
    help="Name of the Python execultable file",
    type=PathLike,
    default=None,
)

simulate_parser.add_argument(
    "-w",
    "--workdir",
    help="Working directory path",
    type=PathLike,
    default=os.getcwd(),
)

simulate_parser.add_argument(
    "-f",
    "--facility",
    help="Facility's name",
    type=str,
    default=None,
)

virtualisation_group = simulate_parser.add_mutually_exclusive_group()

virtualisation_group.add_argument(
    "-v",
    "--venv",
    help="Virtual environment to use for the experiment",
    type=PathLike,
    default=None,
)

virtualisation_group.add_argument(
    "-i",
    "--image",
    help="Path to the Docker/Singularity/Apptainer image",
    type=PathLike,
    default=None,
)

simulate_parser.add_argument(
    "-ws",
    "--world-size",
    help="Number of process to spawn (max. number of available GPUs)",
    type=int,
    default=1,
)

simulate_parser.add_argument(
    "-n",
    "--nodelist",
    help="List of available computing nodes",
    nargs="+",
    default=["localhost"],
)

simulate_parser.add_argument(
    "-args",
    "--arguments",
    help="Command line arguments to be passed to the executable",
    type=str,
    nargs="+",
    default=[],
)

simulate_parser.add_argument(
    "-fs",
    "--federated-scaling",
    help="Enable Federated scaling with the specified federated group size",
    type=str,
    default=None,
)
