"""Argument parser for xFFL
The basic parser offers informative functions.
All the advanced features are offered by the subcommands and the relative subparsers
"""

import argparse
import logging
import os

# Base parser
parser = argparse.ArgumentParser(
    prog="Cross-Facility Federated Learning (xFFL)",
    description="Cross-Facility Federated Learning (xFFL) is a federated learning (FL) framework based on the StreamFlow workflow management system (WMS) developed in the Parallel Computing [Alpha] research group at the University of Turin, Italy.",
    add_help=False,
)

parser.add_argument("-v", "--version", help="Display xFFL version", action="store_true")

parser.add_argument(
    "-h", "--help", help="show this help message and exit", action="store_true"
)

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
subparsers = parser.add_subparsers(dest="command", help="Subcommands help")

# Subcommand: config
config_parser = subparsers.add_parser(
    "config", help="Guided xFFL experiment configuration"
)

config_parser.add_argument(
    "-w",
    "--workdir",
    help="Working directory path",
    type=str,
    default=os.getcwd(),
)
config_parser.add_argument(
    "-p", "--project", help="Project name", type=str, required=True
)
