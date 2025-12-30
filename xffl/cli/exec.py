"""xFFL-handled experiment launching.

This script wraps StreamFlow with a simple Python CLI,
offering a homogeneous interface with xFFL.
"""

import argparse
import importlib.util
import socket
import subprocess
import sys
import time
from logging import Logger, getLogger
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List

import xffl.cli.parser as cli_parser
from xffl.cli.utils import get_facilitator_path
from xffl.custom.types import FileLike, PathLike
from xffl.distributed.networking import get_cells_ids

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


# --------------------------------------------------------------------------- #
#                             Environment Setup                               #
# --------------------------------------------------------------------------- #


def _import_from_path(module_name: str, file_path: FileLike):
    logger.debug(f"Importing {module_name} from {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _setup_env(args: argparse.Namespace, mapping: Dict[str, str]) -> Dict[str, str]:
    """Create a mapping between CLI arguments and environment variables.

    :param args: CLI arguments.
    :type args: Dict[str, Any]
    :param mapping: Mapping between environment variables and CLI argument names.
    :type mapping: Dict[str, str]
    :return: New environment variables dictionary.
    :rtype: Dict[str, str]
    """
    args_dict: Dict[str, Any] = vars(args)
    env: Dict[str, Any] = {
        env_var: str(args_dict[parse_var]) if parse_var in args_dict else None
        for env_var, parse_var in mapping.items()
    }
    if args.image:
        config_module: ModuleType = _import_from_path(
            "configuration", args.configuration
        )
        config = getattr(config_module, args.config)()

        env["XFFL_MODEL_FOLDER"] = config.model.path + config.model.name
        env["XFFL_DATASET_FOLDER"] = config.dataset.path + config.dataset.name
        env["XFFL_CODE_FOLDER"] = str(PathLike(args.executable).parent)

    return env


def _setup_execution_env(args: SimpleNamespace) -> Dict[str, str]:
    """Setup the environment variables for the execution.

    :param args: CLI arguments
    :type args: argparse.Namespace
    :raises ValueError: If no execution environment (venv or image) is provided
    :return: Mapping of environment variables
    :rtype: Dict[str, str]
    """
    base_env_mapping: Dict[str, str] = {
        "XFFL_WORLD_SIZE": "world_size",
        "XFFL_NUM_NODES": "num_nodes",
        "MASTER_ADDR": "masteraddr",
        "XFFL_FACILITY": "facility",
    }

    if args.venv:
        logger.debug("Using virtual environment: %s", args.venv)
        env_mapping = {"XFFL_VENV": "venv"} | base_env_mapping
    elif args.image:
        logger.debug("Using container image: %s", args.image)
        env_mapping = {
            "XFFL_CODE_FOLDER": "workdir",
            "XFFL_MODEL_FOLDER": "model",
            "XFFL_DATASET_FOLDER": "dataset",
            "XFFL_IMAGE": "image",
        } | base_env_mapping
    elif sys.prefix != sys.base_prefix:
        # Already in a venv, use it
        args.venv = sys.prefix
        logger.debug("Using current virtual environment: %s", args.venv)
        env_mapping = {"XFFL_VENV": "venv"} | base_env_mapping
    else:
        raise ValueError("No execution environment specified [container/virtual env]")

    env: Dict[str, Any] = _setup_env(args=args, mapping=env_mapping)
    env["XFFL_EXECUTION"] = "true"

    if args.image:
        exe_path = Path(args.executable)
        env["XFFL_CODE_FOLDER"] = str(exe_path.parent)
        args.executable = exe_path.name

    logger.debug("New local execution xFFL environment variables: %s", env)
    return env


# --------------------------------------------------------------------------- #
#                             Main Execution                                  #
# --------------------------------------------------------------------------- #


def exec(args: argparse.Namespace) -> int:
    """Execute the xFFL execution.

    :param args: CLI arguments
    :type args: argparse.Namespace
    :return: Exit code (0 if success, 1 otherwise)
    :rtype: int
    """

    # Replace localhost with actual hostname
    if args.nodelist == []:
        args.nodelist = [socket.gethostname()]

    args.num_nodes = len(args.nodelist)
    args.masteraddr = args.nodelist[0]
    args.world_size = args.num_nodes * args.processes_per_node

    try:
        env = _setup_execution_env(args=args)
    except ValueError as err:
        logger.error("Failed to setup execution environment: %s", err)
        raise

    facilitator_script = get_facilitator_path()

    # Federated scaling
    if args.federated_scaling is not None:
        if args.federated_scaling == "auto":
            federated_local_size = get_cells_ids(nodes=args.nodelist, cell_size=180)
            if federated_local_size:
                env["XFFL_FEDERATED_LOCAL_WORLD_SIZE"] = str(
                    federated_local_size
                ).translate(str.maketrans("", "", "[] "))
        else:
            env["XFFL_FEDERATED_LOCAL_WORLD_SIZE"] = str(args.federated_scaling)

    # Environment string
    env_str = " ".join(f"{k}={v}" for k, v in env.items())

    logger.info("Running local execution...")
    start_time = time.perf_counter()

    processes: List[subprocess.Popen] = []
    return_code = 0

    try:
        for index, node in enumerate(args.nodelist):
            ssh_command = [
                "ssh",
                "-oStrictHostKeyChecking=no",
                node,
                '"',
                " ".join(
                    [
                        env_str,
                        f"XFFL_NODEID={index}",
                        str(facilitator_script),
                        str(args.executable),
                    ]
                ),
                '"',
            ]
            logger.debug("Execution command on %s: %s", node, " ".join(ssh_command))

            processes.append(
                subprocess.Popen(
                    " ".join(ssh_command),
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    shell=True,
                    universal_newlines=True,
                )
            )

        for process in processes:
            return_code += process.wait()

    except (OSError, ValueError) as exc:
        logger.exception("xFFL execution failed: %s", exc)
        return 1
    else:
        elapsed = time.perf_counter() - start_time
        logger.debug("Total execution time: %.2f seconds", elapsed)

    return 0 if return_code == 0 else 1


def main(args: argparse.Namespace) -> int:
    """Entrypoint for xFFL execution.

    :param args: CLI arguments
    :type args: argparse.Namespace
    :return: Exit code
    :rtype: int
    """
    logger.info("*** Cross-Facility Federated Learning (xFFL) - Execution starting ***")
    try:
        return exec(args=args)
    except Exception as exception:
        logger.exception("Execution failed: %s", exception)
        # raise exception
    finally:
        logger.info(
            "*** Cross-Facility Federated Learning (xFFL) - Execution finished ***"
        )


if __name__ == "__main__":
    main(cli_parser.subparsers.choices["exec"].parse_args())
