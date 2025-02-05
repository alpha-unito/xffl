"""xFFL-handled experiment launching

This script wraps StreamFlow with a simple Python CLI, offering a homogeneous interface with xFFL
"""

import argparse
import inspect
import os
import subprocess
import sys
import time
from logging import Logger, getLogger
from subprocess import PIPE
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
    # Updating the environment with the virtualization technology specified
    env = {}
    if args.venv:
        logger.debug(f"Using virtual environment: {args.venv}")
        # Creating new environment variables based on the provided mapping
        env_mapping = {
            "XFFL_WORLD_SIZE": "world_size",
        }
        env = setup_env(args=vars(args), mapping=env_mapping, parser=simulate_parser)
        env["VENV"] = check_default_value("venv", args.venv, simulate_parser)
    elif args.image:
        logger.debug(f"Using container image: {args.venv}")
        # Creating new environment variables based on the provided mapping
        env_mapping = {
            "CODE_FOLDER": "workdir",
            "MODEL_FOLDER": "model",
            "DATASET_FOLDER": "dataset",
            "XFFL_WORLD_SIZE": "world_size",
        }
        env = setup_env(args=vars(args), mapping=env_mapping, parser=simulate_parser)
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


def simulate(
    args: argparse.Namespace,
) -> None:
    # Environment creation
    env = setup_simulation_env(args=args)
    logger.debug(f"Updated xFFL environment: {env}")

    # Simulation command creation
    facilitator_script = os.path.join(
        os.path.dirname(inspect.getfile(xffl.workflow)), "scripts", "facilitator.sh"
    )
    executable_script = check_default_value(
        "executable", args.executable, simulate_parser
    )

    # Launch facilitator
    logger.info(f"Running local simulation...")
    command = " ".join(
        [facilitator_script, executable_script] + [arg for arg in args.arguments]
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
    from xffl.cli.parser import simulate_parser

    try:
        main(simulate_parser.parse_args())
    except KeyboardInterrupt as e:
        logger.exception(e)
