"""CLI utility methods for xFFL."""

import argparse
import inspect
from logging import Logger, getLogger
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from xffl.custom.types import FileLike, FolderLike, PathLike
from xffl.utils.utils import check_input, get_param_name, resolve_path

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


def check_default_value(
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


def check_cli_arguments(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> SimpleNamespace:
    """Check CLI arguments and expand relative paths into absolute ones.

    :param args: Command line arguments.
    :type args: argparse.Namespace
    :param parser: Command line argument parser.
    :type parser: argparse.ArgumentParser
    :return: Expanded namespace with absolute paths where applicable.
    :rtype: SimpleNamespace
    """

    for key, value in vars(args).items():
        check_default_value(argument_name=key, argument_value=value, parser=parser)

    namespace = vars(args).copy()

    for action in parser._actions:
        if action.type in (FolderLike, FileLike, PathLike):
            param_name = get_param_name(action.option_strings, parser.prefix_chars)
            target_name = (
                param_name
                if param_name in namespace
                else action.dest  # fallback to dest if flags differ
            )

            if namespace.get(target_name):
                namespace[target_name] = str(Path(namespace[target_name]).resolve())

    if "arguments" in namespace:
        namespace["arguments"] = expand_paths_in_args(namespace["arguments"])

    return SimpleNamespace(**namespace)


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


def setup_env(args: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, str]:
    """Create a mapping between CLI arguments and environment variables.

    :param args: CLI arguments.
    :type args: Dict[str, Any]
    :param mapping: Mapping between environment variables and CLI argument names.
    :type mapping: Dict[str, str]
    :return: New environment variables dictionary.
    :rtype: Dict[str, str]
    """
    return {env_var: str(args[parse_var]) for env_var, parse_var in mapping.items()}


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
    base_dir: Path = Path(resolve_path(path=dir_path))

    if not base_dir.exists():
        logger.error(f"The provided working directory path {base_dir} does not exist.")
        raise FileNotFoundError(base_dir)

    target_dir: Path = base_dir / folder_name
    logger.debug(f"Attempting to create directory {target_dir}")

    try:
        target_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        answer = check_input(
            f"Directory {target_dir} already exists. Overwrite it? [y/n]: ",
            "Answer not accepted.",
            lambda ans: ans.lower() in ("y", "yes", "n", "no"),
        )
        if answer.lower() in ("n", "no"):
            raise
    return target_dir.resolve()
