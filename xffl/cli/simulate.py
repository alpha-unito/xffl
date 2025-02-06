"""xFFL-handled experiment launching

This script wraps StreamFlow with a simple Python CLI, offering a homogeneous interface with xFFL
"""

import argparse
import os
import subprocess
import sys
import time
from logging import Logger, getLogger
from typing import Dict

from xffl.cli.parser import simulate_parser
from xffl.cli.utils import check_cli_arguments, get_facilitator_path, setup_env

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def setup_simulation_env(args: argparse.Namespace) -> Dict[str, str]:
    """Sets up the simulation environment variables

    :param args: CLI arguments
    :type args: argparse.Namespace
    :raises ValueError: If not virtualization environment is provided # TODO: make possible to run bare-metal (useful for module)
    :return: Updated environment
    :rtype: Dict[str, str]
    """
    # Creating the environment mapping with the virtualization technology specified
    if args.venv:
        logger.debug(f"Using virtual environment: {args.venv}")
        env_mapping = {"XFFL_WORLD_SIZE": "world_size", "VENV": "venv"}
    elif args.image:
        logger.debug(f"Using container image: {args.venv}")
        env_mapping = {
            "CODE_FOLDER": "workdir",
            "MODEL_FOLDER": "model",
            "DATASET_FOLDER": "dataset",
            "XFFL_WORLD_SIZE": "world_size",
            "IMAGE": "image",
        }
    else:
        raise ValueError("No execution environment specified [container/virtual env]")

    # Creating new environment variables based on the provided mapping
    env = setup_env(args=vars(args), mapping=env_mapping)
    env["FACILITY"] = "local"

    # New environment created - debug logging
    logger.debug(f"New local simulation xFFL environment variables: {env}")

    # Returning the old environment updated
    xffl_env = os.environ.copy()
    xffl_env.update(env)

    return xffl_env


def simulate(
    args: argparse.Namespace,
) -> None:
    # Check the CLI arguments
    check_cli_arguments(args=args, parser=simulate_parser)

    # Environment creation
    try:
        env = setup_simulation_env(args=args)
    except ValueError as e:
        raise e
    logger.debug(f"Updated xFFL environment: {env}")

    # Simulation command creation
    facilitator_script = get_facilitator_path()

    # Launch facilitator
    logger.info(f"Running local simulation...")
    command = " ".join(
        [facilitator_script, args.executable] + [arg for arg in args.arguments]
    )
    logger.debug(f"Local simulation command: {command}")
    start_time = time.perf_counter()
    try:
        return_code = subprocess.Popen(
            command,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            shell=True,
            universal_newlines=True,
        ).wait()
    except (OSError, ValueError) as e:
        logger.exception(e)
        return 1
    else:
        logger.info(
            f"Total simulation execution time: {(time.perf_counter() - start_time):.2f} seconds"
        )

    return return_code


def main(args: argparse.Namespace) -> int:
    """Local script run through xFFL

    :param args: Command line arguments
    :type args: argparse.Namespace
    :return: Exit code
    :rtype: int
    """
    logger.info("*** Cross-Facility Federated Learning (xFFL) - Simulation ***")
    try:
        simulate(args=args)
    except Exception as e:
        logger.exception(e)
        raise
    finally:
        logger.info("*** Cross-Facility Federated Learning (xFFL) - Simulation ***")


if __name__ == "__main__":
    try:
        main(simulate_parser.parse_args())
    except KeyboardInterrupt as e:
        logger.exception(e)
