"""xFFL-handled experiment launching

This script wraps StreamFlow with a simple Python CLI, offering a homogeneous interface with xFFL
"""

import argparse
import subprocess
import sys
import time
from logging import Logger, getLogger
from types import SimpleNamespace
from typing import Dict

from xffl.cli.parser import simulate_parser
from xffl.cli.utils import check_cli_arguments, get_facilitator_path, setup_env
from xffl.distributed.networking import get_cells_ids

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def setup_simulation_env(args: SimpleNamespace) -> Dict[str, str]:
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
        env_mapping = {
            "XFFL_WORLD_SIZE": "world_size",
            "XFFL_NUM_NODES": "num_nodes",
            "MASTER_ADDR": "masteraddr",
            "VENV": "venv",
            "XFFL_FACILITY": "facility",
        }
    elif args.image:
        logger.debug(f"Using container image: {args.venv}")
        env_mapping = {
            "CODE_FOLDER": "workdir",
            "MODEL_FOLDER": "model",
            "DATASET_FOLDER": "dataset",
            "XFFL_WORLD_SIZE": "world_size",
            "XFFL_NUM_NODES": "num_nodes",
            "MASTER_ADDR": "masteraddr",
            "IMAGE": "image",
            "XFFL_FACILITY": "facility",
        }
    else:
        raise ValueError("No execution environment specified [container/virtual env]")

    # Creating new environment variables based on the provided mapping
    env = setup_env(args=vars(args), mapping=env_mapping)
    env["XFFL_SIMULATION"] = "true"

    # New environment created - return it as a string
    logger.debug(f"New local simulation xFFL environment variables: {env}")

    return env


def simulate(
    args: argparse.Namespace,
) -> int:
    # Check the CLI arguments
    args = check_cli_arguments(args=args, parser=simulate_parser)
    args.num_nodes = len(args.nodelist)
    args.masteraddr = args.nodelist[0]
    args.world_size = args.num_nodes * args.processes_per_node

    # Environment creation
    try:
        env = setup_simulation_env(args=args)
    except ValueError as err:
        raise err
    logger.debug(f"Updated xFFL environment: {env}")

    # Simulation command creation
    facilitator_script = get_facilitator_path()

    # Nodes cell IDs calculation for the FederatedScaling feature
    if args.federated_scaling is not None:
        if args.federated_scaling == "auto":
            federated_local_size = get_cells_ids(nodes=args.nodelist, cell_size=180)
            if federated_local_size:
                env["XFFL_FEDERATED_LOCAL_WORLD_SIZE"] = (
                    str(federated_local_size)
                    .replace("]", "")
                    .replace("[", "")
                    .replace(" ", "")
                )
        else:
            env["XFFL_FEDERATED_LOCAL_WORLD_SIZE"] = args.federated_scaling

    env_str = ""
    for key in env:
        env_str += f"{key}={env[key]} "

    # Launch facilitator
    logger.info(f"Running local simulation...")
    start_time = time.perf_counter()
    try:
        processes = []
        return_code = 0

        for index, node in enumerate(args.nodelist):
            command = (
                [
                    "ssh",
                    "-oStrictHostKeyChecking=no",
                    node,
                    '"',
                    env_str,
                    f"XFFL_NODEID={index}",
                    facilitator_script,
                    args.executable,
                ]
                + [arg for arg in args.arguments]
                + ['"']
            )
            command_str = " ".join(command)
            logger.debug(f"Simulation command on {node}: {command_str}")

            processes.append(
                subprocess.Popen(
                    command_str,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    shell=True,
                    universal_newlines=True,
                )
            )

        for process in processes:
            return_code += process.wait()

    except (OSError, ValueError) as exception:
        logger.exception(exception)
        return 1
    else:
        logger.info(
            f"Total simulation execution time: {(time.perf_counter() - start_time):.2f} seconds"
        )

    return 0 if return_code == 0 else 1


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
    except Exception as exception:
        logger.exception(exception)
        raise exception
    finally:
        logger.info("*** Cross-Facility Federated Learning (xFFL) - Simulation ***")
        return 0


if __name__ == "__main__":
    try:
        main(simulate_parser.parse_args())
    except KeyboardInterrupt as e:
        logger.exception(e)
