"""Argument parser for xFFL

The basic parser offers informative functions.
All the advanced features are offered by the subcommands and the relative subparsers
"""

import argparse
import logging
import os

from xffl.custom.formatter import CustomFormatter

# Base parser
parser = argparse.ArgumentParser(
    prog="xffl",
    description="Cross-Facility Federated Learning (xFFL) is a federated learning (FL) framework based on the StreamFlow workflow management system (WMS) developed in the Parallel Computing [Alpha] research group at the University of Turin, Italy.",
    add_help=False,
    conflict_handler="resolve",
)

parser.add_argument(
    "-h", "--help", help="show this help message and exit", action="store_true"
)

parser.add_argument("-v", "--version", help="Display xFFL version", action="store_true")

parser.add_argument(
    "-d",
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
    "-h", "--help", help="show this help message and exit", action="store_true"
)

config_parser.add_argument("-p", "--project", help="Project name [REQUIRED]", type=str)

config_parser.add_argument(
    "-w",
    "--workdir",
    help="Working directory path",
    type=str,
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
    "-h", "--help", help="show this help message and exit", action="store_true"
)

run_parser.add_argument(
    "-w",
    "--workdir",
    help="Working directory path",
    type=str,
    default=os.getcwd(),
)

run_parser.add_argument(
    "--name",
    nargs="?",
    type=str,
    help="Name of the current workflow. Will be used for search and indexing",
)

run_parser.add_argument("-p", "--project", help="Project name", type=str)

run_parser.add_argument(
    "--outdir",
    default=os.getcwd(),
    type=str,
    help="Output directory in which to store final results of the workflow (default: current directory)",
)

run_parser.add_argument(
    "--quiet", action="store_true", help="Only prints results, warnings and errors"
)
