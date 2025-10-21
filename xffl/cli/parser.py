"""Argument parser for xFFL.

This module defines the main parser and all subcommands for the xFFL CLI.

The base parser offers common options like version and debug logging.
Advanced features are provided by subcommands and their specific options.
"""

import argparse
import logging
import os
import subprocess
from typing import Tuple

from xffl.custom.types import FileLike, FolderLike


def _add_common_project_options(subparser: argparse.ArgumentParser) -> None:
    """Add common project-related options to a subparser.

    :param subparser: The argparse subparser to extend
    :type subparser: argparse.ArgumentParser
    """
    subparser.add_argument(
        "-p",
        "--project",
        help="Name of the project or experiment. Default is 'project'.",
        type=str,
        default="project",
    )
    subparser.add_argument(
        "-w",
        "--workdir",
        help="Working directory where the experiment files are stored. "
        "Defaults to the current working directory.",
        type=FolderLike,
        default=os.getcwd(),
    )


def _get_default_nodelis() -> Tuple[str]:
    return (
        tuple(
            subprocess.run(
                ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
                capture_output=True,
                text=True,
            ).stdout.split("\n")[:-1]
        )
        if "SLURM_JOB_NODELIST" in os.environ
        else None
    )


def _get_default_ppn() -> int:
    return (
        len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        if "CUDA_VISIBLE_DEVICES" in os.environ
        else len(os.environ["ROCR_VISIBLE_DEVICES"].split(","))
    )


def build_parser() -> Tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    """Build the main xFFL argument parser.

    :return: Configured argparse parser
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="xffl",
        description=(
            "Cross-Facility Federated Learning (xFFL) is a federated learning (FL) "
            "framework based on the StreamFlow workflow management system (WMS), "
            "developed by the Parallel Computing [Alpha] research group at the "
            "University of Turin, Italy."
        ),
    )

    # Global options
    parser.add_argument(
        "-v",
        "--version",
        help="Display the current version of xFFL and exit.",
        action="store_true",
    )
    parser.add_argument(
        "-dbg",
        "--debug",
        help="Enable debug logging to show detailed messages for troubleshooting. "
        "By default, logging is set to INFO level.",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.INFO,
    )

    # Subparsers
    subparsers = parser.add_subparsers(
        dest="command", help="Choose one of the available xFFL subcommands to execute."
    )

    # Subcommand: config
    config_parser = subparsers.add_parser(
        "config",
        description="Interactively configure a new xFFL experiment.",
        help="Create or edit a federated learning experiment configuration.",
    )
    _add_common_project_options(config_parser)

    # Subcommand: run
    run_parser = subparsers.add_parser(
        "run",
        description="Run a previously configured xFFL experiment.",
        help="Execute an experiment using xFFL.",
    )
    _add_common_project_options(run_parser)

    run_parser.add_argument(
        "-o",
        "--outdir",
        help="Directory where experiment results will be stored. "
        "If not specified, a default location within the working directory is used.",
        type=str,
        default=None,
    )
    run_parser.add_argument(
        "--quiet",
        help="Suppress detailed logs and show only warnings, errors, and final results.",
        action="store_true",
    )
    run_parser.add_argument(
        "--validate",
        help="Check the validity of StreamFlow workflow documents before running.",
        action="store_true",
    )

    # Subcommand: exec
    exec_parser = subparsers.add_parser(
        "exec",
        description="Execute a Python script or experiment locally through xFFL.",
        help="Run a local script with xFFL execution framework.",
    )

    exec_parser.add_argument(
        "executable",
        help="Path to the Python script or executable to run.",
        type=FileLike,
    )

    exec_parser.add_argument(
        "configuration",
        help="Path to the run configuration file.",
        type=FileLike,
    )

    exec_parser.add_argument(
        "-c",
        "--config",
        help="Desired configuration to be instantiated from the configuration file.",
        type=str,
        default="xffl_config",
    )

    exec_parser.add_argument(
        "-f",
        "--facility",
        help="Name of the computational facility to use. Default is 'leonardo'.",
        type=str,
        default="leonardo",
    )

    exec_parser.add_argument(
        "-fs",
        "--federated-scaling",
        help="Enable federated scaling and specify the size of the federated group.",
        type=str,
        default=None,
    )

    # Mutually exclusive group for virtualization options
    virtualization_group = exec_parser.add_mutually_exclusive_group()
    virtualization_group.add_argument(
        "-v",
        "--venv",
        help="Use the specified Python virtual environment for the experiment.",
        type=FolderLike,
        default=None,
    )
    virtualization_group.add_argument(
        "-i",
        "--image",
        help="Use a Docker, Singularity, or Apptainer image for the execution.",
        type=FileLike,
        default=None,
    )

    exec_parser.add_argument(
        "-n",
        "--nodelist",
        help="List of compute nodes available for the execution. Default is ['localhost'].",
        nargs="+",
        default=_get_default_nodelis(),
    )

    exec_parser.add_argument(
        "-ppn",
        "--processes-per-node",
        help="Number of GPUs or processes available per compute node. Default is 1.",
        type=int,
        default=_get_default_ppn(),
    )

    return parser, subparsers


# Export the parser instance
parser, subparsers = build_parser()
"""The main xFFL argument parser instance."""
