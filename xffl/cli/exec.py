"""xFFL-handled experiment launching.

This script wraps StreamFlow with a simple Python CLI,
offering a homogeneous interface with xFFL.
"""

import importlib.util
import inspect
import subprocess
import sys
import time
from argparse import Namespace
from importlib.machinery import ModuleSpec
from logging import Logger, getLogger
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import xffl.cli.parser as cli_parser
from xffl.custom.config_info import XFFLConfig
from xffl.custom.types import FileLike, PathLike
from xffl.distributed.networking import get_cells_ids

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


# --------------------------------------------------------------------------- #
#                             Environment Setup                               #
# --------------------------------------------------------------------------- #


def _import_from_path(module_name: str, file_path: FileLike) -> Optional[ModuleType]:
    """Dynamically imports a module from a file path

    :param module_name: Name of the module to import
    :type module_name: str
    :param file_path: Path to the module's file
    :type file_path: FileLike
    :return: The imported module
    :rtype: Optional[ModuleType]
    """
    logger.debug(f"Importing {module_name} from {file_path}")
    spec: Optional[ModuleSpec] = importlib.util.spec_from_file_location(
        module_name, str(file_path)
    )
    module: Optional[ModuleType] = None
    if spec is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        if spec.loader is not None:
            spec.loader.exec_module(module)
        else:
            logger.warning(
                f"Impossible to load {module_name} from {file_path}. Loading interrupted."
            )
    else:
        logger.warning(
            f"{module_name} from {file_path} not found. Loading interrupted."
        )
    return module


def _setup_env(
    args: SimpleNamespace | Namespace, mapping: Dict[str, str]
) -> Dict[str, Any]:
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
        config_module: Optional[ModuleType] = _import_from_path(
            "configuration", args.configuration
        )
        if config_module is not None:
            config: XFFLConfig = getattr(config_module, args.config)()

            env["XFFL_MODEL_FOLDER"] = config.model.path + config.model.name
            env["XFFL_DATASET_FOLDER"] = config.dataset.path + config.dataset.name
            env["XFFL_CODE_FOLDER"] = str(PathLike(args.executable).parent)

    return env


def _setup_execution_env(args: SimpleNamespace | Namespace) -> Dict[str, str]:
    """Setup the environment variables for the execution.

    :param args: CLI arguments
    :type args: Namespace
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
        env_mapping: Dict[str, str] = {"XFFL_VENV": "venv"} | base_env_mapping
    elif args.image:
        logger.debug("Using container image: %s", args.image)
        env_mapping: Dict[str, str] = {
            "XFFL_CODE_FOLDER": "workdir",
            "XFFL_MODEL_FOLDER": "model",
            "XFFL_DATASET_FOLDER": "dataset",
            "XFFL_IMAGE": "image",
        } | base_env_mapping
    elif sys.prefix != sys.base_prefix:
        # Already in a venv, use it
        args.venv = sys.prefix
        logger.debug("Using current virtual environment: %s", args.venv)
        env_mapping: Dict[str, str] = {"XFFL_VENV": "venv"} | base_env_mapping
    else:
        raise ValueError("No execution environment specified [container/virtual env]")

    env: Dict[str, Any] = _setup_env(args=args, mapping=env_mapping)
    env["XFFL_EXECUTION"] = "true"

    if args.image:
        env["XFFL_CODE_FOLDER"] = args.executable.parent
        args.executable = args.executable.name

    logger.debug("New local execution xFFL environment variables: %s", env)
    return env


def _get_facilitator_path() -> FileLike:
    """Return the absolute path of the facilitator script.

    :return: Facilitator absolute file path.
    :rtype: Path
    """
    import xffl.workflow

    workflow_dir: Path = Path(inspect.getfile(xffl.workflow)).parent
    return FileLike(workflow_dir / "scripts" / "facilitator.sh")


# --------------------------------------------------------------------------- #
#                             Main Execution                                  #
# --------------------------------------------------------------------------- #


def exec(args: Namespace) -> int:
    """Execute the xFFL execution.

    :param args: CLI arguments
    :type args: Namespace
    :return: Exit code (0 if success, 1 otherwise)
    :rtype: int
    """

    args.num_nodes = len(args.nodelist)
    args.masteraddr = args.nodelist[0]
    args.world_size = args.num_nodes * args.processes_per_node

    try:
        env: Dict[str, Any] = _setup_execution_env(args=args)
    except ValueError as err:
        logger.error("Failed to setup execution environment: %s", err)
        raise

    facilitator_script: FileLike = _get_facilitator_path()

    # Federated scaling
    if args.federated_scaling is not None:
        if args.federated_scaling == "auto":
            federated_local_size: Tuple[int, ...] = get_cells_ids(
                nodes=args.nodelist, cell_size=180
            )
            if federated_local_size:
                env["XFFL_FEDERATED_LOCAL_WORLD_SIZE"] = str(
                    federated_local_size
                ).translate(str.maketrans("", "", "[] "))
        else:
            env["XFFL_FEDERATED_LOCAL_WORLD_SIZE"] = str(args.federated_scaling)

    env_str: str = " ".join(f"{k}={v}" for k, v in env.items())

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


def main(args: Namespace) -> int:
    """Entrypoint for xFFL execution.

    :param args: CLI arguments
    :type args: Namespace
    :return: Exit code
    :rtype: int
    """
    logger.info("*** Cross-Facility Federated Learning (xFFL) - Execution starting ***")
    try:
        return exec(args=args)
    except Exception as exception:
        logger.exception("Execution failed: %s", exception)
        return 1
    finally:
        logger.info(
            "*** Cross-Facility Federated Learning (xFFL) - Execution finished ***"
        )


if __name__ == "__main__":
    main(cli_parser.subparsers.choices["exec"].parse_args())
