"""xFFL-handled experiment launching

This script wraps StreamFlow with a simple Python CLI, offering a homogeneous interface with xFFL
"""

import argparse
import inspect
import logging
import os
import subprocess
from logging import Logger, getLogger
from typing import Dict

import xffl
from xffl.cli.parser import simulate_parser
from xffl.cli.utils import check_default_value, setup_env

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def setup_simulation_env(args: argparse.Namespace) -> Dict[str, str]:
    """Sets up the simulation environment variables

    :param args: CLI arguments
    :type args: argparse.Namespace
    :return: Updated environment
    :rtype: Dict[str, str]
    """
    # Creating a new environment based on the provided mapping
    env_mapping = {
        "CODE_FOLDER": "workdir",
        "MODEL_FOLDER": "model",
        "DATASET_FOLDER": "dataset",
        "XFFL_WORLD_SIZE": "world_size",
    }
    env = setup_env(args=vars(args), mapping=env_mapping, parser=simulate_parser)

    # Updating the new environment with the virtualization technology specified
    if args.venv:
        logger.debug(f"Using virtual environment: {args.venv}")
        env["VENV"] = check_default_value("venv", args.venv, simulate_parser)
    elif args.image:
        logger.debug(f"Using container image: {args.venv}")
        env["IMAGE"] = check_default_value("image", args.image, simulate_parser)
    else:
        logger.error(f"No execution environment specified [container/virtual env]")

    # Specifying the local execution
    env["FACILITY"] = "local"

    # New environment created - debug logging
    logger.debug(f"New local simulation xFFL environment variables: {env}")

    # Returning the old environment updated
    xffl_env = os.environ.copy()
    xffl_env.update(env)
    return xffl_env


def simulate(args: argparse.Namespace) -> None:
    # Environment creation
    env = setup_simulation_env(args=args)
    logger.debug(f"Updated xFFL environment: {env}")

    # Simulation command creation
    facilitator_script = os.path.join(
        os.path.dirname(inspect.getfile(xffl.workflow)), "scripts", "facilitator.sh"
    )
    debug_mode = "-dbg" if args.loglevel == logging.DEBUG else ""

    executable_script = check_default_value(
        "executable", args.executable, simulate_parser
    )
    model_path = check_default_value("model", args.model, simulate_parser)
    dataset_path = check_default_value("dataset", args.dataset, simulate_parser)

    command = [
        facilitator_script,
        executable_script,
        debug_mode,
        "-m" if model_path else "",
        model_path,
        "-d" if dataset_path else "",
        dataset_path,
    ]
    logger.debug(f"Local simulation command: {' '.join(command)}")

    # Launch facilitator
    logger.info(f"Running local simulation...")
    process = subprocess.run(command, env=env)

    return process.returncode


def main(args: argparse.Namespace) -> int:
    """Local script run through xFFL

    :param args: Command line arguments
    :type args: argparse.Namespace
    :return: Exit code
    :rtype: int
    """
    logger.info("*** Cross-Facility Federated Learning (xFFL) - Simulation ***")
    exit_code = 0
    try:
        simulate(args=args)
    except Exception as e:
        logger.exception(e)
        exit_code = 1
    finally:
        logger.info("*** Cross-Facility Federated Learning (xFFL) - Simulation ***")
        return exit_code


if __name__ == "__main__":
    from xffl.cli.parser import simulate_parser

    try:
        main(simulate_parser.parse_args())
    except KeyboardInterrupt as e:
        logger.exception(e)
