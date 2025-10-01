"""Argument parser for xFFL.

This module defines the main parser and all subcommands for the xFFL CLI.

The base parser offers common options like version and debug logging.
Advanced features are provided by subcommands and their specific options.
"""

import argparse
import logging
import os

from xffl.custom.types import PathLike


def _add_common_project_options(subparser: argparse.ArgumentParser) -> None:
    """Add common project-related options to a subparser.

    :param subparser: The argparse subparser to extend
    :type subparser: argparse.ArgumentParser
    """
    subparser.add_argument(
        "-p", "--project", help="Project name", type=str, default="project"
    )
    subparser.add_argument(
        "-w",
        "--workdir",
        help="Working directory path",
        type=PathLike,
        default=os.getcwd(),
    )


def _add_arguments_option(subparser: argparse.ArgumentParser) -> None:
    """Add the --arguments passthrough option to a subparser.

    :param subparser: The argparse subparser to extend
    :type subparser: argparse.ArgumentParser
    """
    subparser.add_argument(
        "-args",
        "--arguments",
        help="Command line arguments to be passed to the executable",
        type=str,
        nargs="+",
        default=[],
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the main xFFL argument parser.

    :return: Configured argparse parser
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="xffl",
        description=(
            "Cross-Facility Federated Learning (xFFL) is a federated learning (FL) "
            "framework based on the StreamFlow workflow management system (WMS) "
            "developed in the Parallel Computing [Alpha] research group at the "
            "University of Turin, Italy."
        ),
    )

    # Global options
    parser.add_argument(
        "-v", "--version", help="Display xFFL version", action="store_true"
    )
    parser.add_argument(
        "-dbg",
        "--debug",
        help="Enable debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.INFO,
    )

    # Subparsers
    subparsers = parser.add_subparsers(
        dest="command", help="Available xFFL subcommands"
    )

    # Subcommand: config
    config_parser = subparsers.add_parser(
        "config",
        description="Guided xFFL experiment configuration",
        help="Configure an experiment",
    )
    _add_common_project_options(config_parser)

    # Subcommand: run
    run_parser = subparsers.add_parser(
        "run", description="Run an xFFL experiment", help="Run an experiment"
    )
    _add_common_project_options(run_parser)
    _add_arguments_option(run_parser)

    run_parser.add_argument(
        "-o", "--outdir", help="Output directory", type=str, default=None
    )
    run_parser.add_argument(
        "--quiet", help="Only print results, warnings and errors", action="store_true"
    )
    run_parser.add_argument(
        "--validate", help="Validate StreamFlow documents", action="store_true"
    )

    # Subcommand: exec
    exec_parser = subparsers.add_parser(
        "exec", description="Run a script locally through xFFL", help="Execute a script"
    )
    _add_arguments_option(exec_parser)

    exec_parser.add_argument(
        "executable", help="Python executable file to run", type=PathLike
    )
    exec_parser.add_argument(
        "-f", "--facility", help="Facility name", type=str, default="leonardo"
    )
    exec_parser.add_argument(
        "-n",
        "--nodelist",
        help="List of available compute nodes",
        nargs="+",
        default=["localhost"],
    )
    exec_parser.add_argument(
        "-fs",
        "--federated-scaling",
        help="Enable Federated Scaling with the specified group size",
        type=str,
        default=None,
    )

    # Mutually exclusive group for virtualization
    virtualization_group = exec_parser.add_mutually_exclusive_group()
    virtualization_group.add_argument(
        "-v", "--venv", help="Virtual environment path", type=PathLike, default=None
    )
    virtualization_group.add_argument(
        "-i",
        "--image",
        help="Docker/Singularity/Apptainer image path",
        type=PathLike,
        default=None,
    )

    exec_parser.add_argument(
        "-p",
        "--processes-per-node",
        help="Number of GPUs available per compute node",
        type=int,
        default=1,
    )

    return parser


# Export the parser instance
parser: argparse.ArgumentParser = build_parser()
