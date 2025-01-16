"""Argument parser for xFFL
The basic parser offers informative functions.
All the advanced features are offered by the subcommands and the relative subparsers
"""

import argparse
import os

# Base parser
parser = argparse.ArgumentParser(
    prog="Cross-Facility Federated Learning (xFFL)",
    description="Cross-Facility Federated Learning (xFFL) is a federated learning (FL) framework based on the StreamFlow workflow management system (WMS) developed in the Parallel Computing [Alpha] research group at the University of Turin, Italy.",
)

parser.add_argument("-v", "--version", help="Display xFFL version", action="store_true")

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
config_parser.add_argument(
    "-v", "--verbose", help="Increase verbosity level", action="store_true"
)
