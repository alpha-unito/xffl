"""CLI utility methods for xFFL."""

import argparse
import inspect
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, List

from xffl.custom.types import FileLike, FolderLike, PathLike
from xffl.utils.utils import check_input

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def get_facilitator_path() -> FileLike:
    """Return the absolute path of the facilitator script.

    :return: Facilitator absolute file path.
    :rtype: FileLike
    """
    import xffl.workflow

    workflow_dir = Path(inspect.getfile(xffl.workflow)).parent
    return workflow_dir / "scripts" / "facilitator.sh"


def _check_default_value(
    argument_name: str, argument_value: Any, parser: argparse.ArgumentParser
) -> None:
    """Check if a CLI argument value equals its default.

    :param argument_name: Variable name of the argument.
    :type argument_name: str
    :param argument_value: Actual value of the argument.
    :type argument_value: Any
    :param parser: Parser from which the argument originated.
    :type parser: argparse.ArgumentParser
    """
    default_value = parser.get_default(dest=argument_name)
    if argument_value == default_value:
        logger.debug(
            f'CLI argument "{argument_name}" has default value "{default_value}"'
        )


def expand_paths_in_args(args: List[str], prefix: str = "-") -> List[str]:
    """Expand relative paths in arguments to absolute paths.

    Used when the parser does not define a specific type.

    :param args: List of command-line arguments.
    :type args: List[str]
    :param prefix: Prefix symbol preceding a flag (default: "-").
    :type prefix: str, optional
    :return: List of command-line arguments with expanded paths.
    :rtype: List[str]
    """
    expanded_args: List[str] = []

    for arg in args:
        if not arg.startswith(prefix) and Path(arg).exists():
            expanded_args.append(str(Path(arg).resolve()))
        else:
            expanded_args.append(arg)

    return expanded_args


def check_and_create_dir(dir_path: FolderLike, folder_name: PathLike) -> FolderLike:
    """Check the base directory and create a subfolder.

    :param dir_path: Base directory path.
    :type dir_path: FolderLike
    :param folder_name: Name of the subfolder to create.
    :type folder_name: PathLike
    :raises FileNotFoundError: If the base directory path does not exist.
    :raises FileExistsError: If the target directory already exists and overwrite is denied.
    :return: Absolute path to the created (or existing) folder.
    :rtype: FolderLike
    """

    if not dir_path.exists():
        logger.error(f"The provided working directory path {dir_path} does not exist.")
        raise FileNotFoundError(dir_path)

    target_dir: Path = dir_path / folder_name
    logger.debug(f"Attempting to create directory {target_dir}")

    try:
        target_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        answer = check_input(
            f"Directory {target_dir} already exists. Overwrite it? [y/n]: ",
            "Answer not accepted.",
            lambda reply: reply.lower() in ("y", "yes", "n", "no"),
        )
        if answer.lower() in ("n", "no"):
            raise
    return target_dir.resolve()
