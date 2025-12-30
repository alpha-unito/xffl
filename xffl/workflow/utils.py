"""Utility methods for the workflow configuration creation"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from argparse import _HelpAction, _StoreConstAction
from logging import Logger, getLogger
from types import MappingProxyType
from typing import Any, Final, List, MutableMapping, Tuple

from xffl.custom.types import FileLike, FolderLike
from xffl.utils.utils import get_param_flag, get_param_name

logger: Logger = getLogger(__name__)


CWL_TYPE_MAPPING: Final[MappingProxyType[Any, str]] = MappingProxyType(
    {
        str: "string",
        int: "int",
        float: "float",
        bool: "boolean",
        None: "boolean",
        FolderLike: "Directory",
        FileLike: "File",
    }
)
"""An immutable dictionary mapping Python to CWL types"""


def from_args_to_cwl(
    parser: argparse.ArgumentParser, arguments: List[str]
) -> Tuple[
    MutableMapping[str, Any], MutableMapping[str, str], MutableMapping[str, Any]
]:
    """Converts a Python ArgumentParser into valid dictionaries of CWL inputs entries

    :param parser: Python command line argument parser
    :type parser: argparse.ArgumentParser
    :param arguments: Command line arguments
    :type arguments: List[str]
    :raises (argparse.ArgumentError, argparse.ArgumentTypeError): Argument parsing exceptions
    :return: Three dictionaries with the arguments name as keys and different values: CWL input bidding, CWL type, CWL value
    :rtype: Tuple[MutableMapping[str, Any], MutableMapping[str, str], MutableMapping[str, Any]]
    """
    # Three dictionaries are produced, with three different mappings
    arg_to_bidding, arg_to_type, arg_to_value = {}, {}, {}

    # Parse the command line arguments and convert the namespace to a dictionary
    try:
        namespace = vars(parser.parse_args(arguments))
        logger.debug(f"Script namespace: {namespace}")
    except (SystemExit, argparse.ArgumentError, argparse.ArgumentTypeError) as e:
        raise e

    # Iterate over the parser's declared arguments and compile the three dictionaries
    for action in parser._actions:
        if not isinstance(action, _HelpAction):
            # Convert the action attributes into useful parameter information
            input_ = get_param_name(action.option_strings, parser.prefix_chars)
            cwl_type = CWL_TYPE_MAPPING[action.type]
            required = action.required

            # Model does not need to be added to the configurations, it is handled separately
            if input_ not in ("model", "dataset"):
                # Argument name to CWL input bidding format
                arg_to_bidding[input_] = {
                    "type": cwl_type + ("" if required else "?"),
                } | (
                    {
                        "prefix": get_param_flag(action.option_strings),
                    }
                    if input_ != "workspace"
                    else {}
                )

                # TODO: this does not seems to be working with dbg store_true
                if add_default_value := isinstance(
                    action.default, bool
                ) or not isinstance(action, _StoreConstAction):
                    arg_to_bidding[input_]["default"] = action.default
                else:
                    logger.warning(
                        f"Default value {action.default} NOT assigned to {input_}"
                    )
                # Argument name to CWL type
                arg_to_type[input_] = cwl_type + ("" if required else "?")

                # Argument name to value (Directory and folder require different format)
                namespace_input = (
                    input_
                    if input_ in namespace
                    else action.dest  # Arguments can be stored in a variable with a name different from their flag, namely dest
                )
                if add_default_value:
                    if isinstance(namespace[namespace_input], str):
                        in_value = namespace[namespace_input].replace(" ", "_")
                    else:
                        in_value = namespace[namespace_input]
                    arg_to_value[input_] = (
                        in_value
                        if action.type not in [FolderLike, FileLike]
                        else (
                            {
                                "class": cwl_type,
                                "path": in_value,
                            }
                            if in_value is not None
                            else None
                        )
                    )
    return arg_to_bidding, arg_to_type, arg_to_value


def import_from_path(module_name: str, file_path: FileLike) -> types.ModuleType:
    """Dynamically import a module from a file

    :param module_name: Name of the module to be imported
    :type module_name: str
    :param file_path: Absolute path to the file containing the module
    :type file_path: FileLike
    :return: Imported Python module
    :rtype: types.ModuleType
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module
