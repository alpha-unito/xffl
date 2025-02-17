"""CLI utility methods"""

import argparse
import inspect
import os
from logging import Logger, getLogger
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import xffl
from xffl.custom.types import FileLike, FolderLike, PathLike
from xffl.utils.utils import get_param_name, resolve_path

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def get_facilitator_path() -> FileLike:
    """Returns the facilitator absolute file path

    :return: Facilitator absolute file path
    :rtype: FileLike
    """
    return os.path.join(
        os.path.dirname(inspect.getfile(xffl.workflow)), "scripts", "facilitator.sh"
    )


def check_default_value(
    argument_name: str, argument_value: Any, parser: argparse.ArgumentParser
) -> None:
    """Checks if a cli argument value equals its default value

    :param argument_name: Variable name of the argument
    :type argument_name: str
    :param argument_value: Actual value of the argument
    :type argument_value: Any
    :param parser: Parser from which the argument originated
    :type parser: argparse.ArgumentParser
    """
    deafult_value = parser.get_default(dest=argument_name)
    if argument_value == deafult_value:
        logger.warning(
            f'CLI argument "{argument_name}" has got default value "{deafult_value}"'
        )


def check_cli_arguments(  # TODO: add checks on Path, File and Folder objects so that is not done elsewhere
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> SimpleNamespace:
    """Checks which cli arguments have the default value and expands relative path into absolte ones

    :param args: Command line arguments
    :type args: argparse.Namespace
    :param parser: Command line argument parser
    :type parser: argparse.ArgumentParser
    """
    for key, value in vars(args).items():
        check_default_value(argument_name=key, argument_value=value, parser=parser)

    namespace = vars(args)
    for action in parser._actions:
        if action.type in [FolderLike, FileLike, PathLike]:
            namespace_input = (
                get_param_name(action.option_strings, parser.prefix_chars)
                if input in namespace
                else action.dest  # Argmuents can be stored in a variable with a name different from their flag, namely dest
            )
            namespace[namespace_input] = (
                os.path.abspath(namespace[namespace_input])
                if namespace[namespace_input]
                else None
            )
    if "arguments" in namespace:
        namespace["arguments"] = expand_paths_in_args(namespace["arguments"])

    return SimpleNamespace(**namespace)


def expand_paths_in_args(args: List[str], prefix: str = "-") -> List[str]:
    """Exapnds relative paths in absolute one when variable type is not available

    :param args: List of command-line arguments
    :type args: List[str]
    :param prefix: Prefix symbol preceding a flag, defaults to "-"
    :type prefix: str, optional
    :raises FileExistsError: If the file path does not exists
    :raises FileNotFoundError: If the file path is not found
    :return: List of command-line arguments with expanded paths
    :rtype: List[str]
    """
    for index, arg in enumerate(args):
        if arg[0] != prefix:
            if os.path.isfile(arg) or os.path.isdir(arg) or os.path.exists(arg):
                if not os.path.isabs(arg):
                    args[index] = os.path.abspath(arg)

    return args


def setup_env(args: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, str]:
    """Creates a mapping between the CLI arguments and new enviromental variables

    :param args: CLI arguments
    :type args: Dict[str, Any]
    :param mapping: Mapping between environmental variables and CLI arguments names
    :type mapping: Dict[str, str]
    :param parser: Parser from which the arguments originated
    :type parser: ArgumentParser
    :return: New environment variables dictionary
    :rtype: Dict[str, str]
    """
    env = {}
    for env_var, parse_var in mapping.items():
        env[env_var] = str(args[parse_var])

    return env


def check_and_create_workdir(
    workdir_path: FolderLike, project_name: PathLike  # TODO: Project name str or path?
) -> FolderLike:
    """Checks the working directory path, the project name, and created the project folder accordingly

    :param workdir_path: Working directory path
    :type workdir_path: FolderLike
    :param project_name: Project name
    :type project_name: PathLike
    :raises FileExistsError: If the project folder already exists
    :raises FileNotFoundError: Il the working directory path is not found
    :return: Absolute path to the project folder
    :rtype: FolderLike
    """
    # Working directory path resolving
    workdir: FolderLike = resolve_path(path=workdir_path)

    if Path(workdir).exists():
        workdir: FolderLike = os.path.join(workdir, project_name)

        logger.debug(f"Attempting to create project directory {workdir}")
        try:
            os.makedirs(workdir)
        except FileExistsError as e:
            raise FileExistsError(f"Project directory {workdir} already exists") from e

    else:
        raise FileNotFoundError(
            f"The provided working directory path {workdir} does not exists"
        )

    return workdir
