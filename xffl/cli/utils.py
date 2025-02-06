"""CLI utility methods"""

import argparse
import inspect
import os
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Dict

import xffl
from xffl.custom.types import FileLike, FolderLike, PathLike
from xffl.utils.utils import resolve_path

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
) -> None:
    """Checks which cli arguments have the default value

    :param args: Command line arguments
    :type args: argparse.Namespace
    :param parser: Command line argument parser
    :type parser: argparse.ArgumentParser
    """
    for key, value in vars(args).items():
        check_default_value(argument_name=key, argument_value=value, parser=parser)


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
