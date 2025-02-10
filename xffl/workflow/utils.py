"""Utility methods for the workflow configuration creation"""

import argparse
import importlib.util
import sys
import types
from argparse import _HelpAction
from types import MappingProxyType
from typing import Any, Final, List, MutableMapping, Tuple

from xffl.custom.types import FileLike, FolderLike


def get_param_flag(list: List[str]) -> str:
    """Gets the full command line parameter flag

    :param list: List of the parameter's flags
    :type list: List[str]
    :return: The full parameter flag
    :rtype: str
    """
    return max(list, key=len)


def get_param_name(list: List[str], prefix: str = "-") -> str:
    """Returns a the command line parameter full name given its flag list

     This method also replaces scores with underscores

    :param list: List of the parameter's flags
    :type list: List[str]
    :param prefix: Prefix symbol preceding a flag, defaults to "-"
    :type prefix: str
    :return: Full parameter name
    :rtype: str
    """

    return get_param_flag(list=list).lstrip("-").replace("-", "_")


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
    parser: argparse.ArgumentParser, arguments: List[str], start_position: int = 0
) -> Tuple[
    MutableMapping[str, Any], MutableMapping[str, str], MutableMapping[str, Any]
]:
    """Converts a Python ArgumentParser into valid dictionaries of CWL inputs entries

    :param parser: Python command line argument parser
    :type parser: argparse.ArgumentParser
    :param arguments: Command line arguments
    :type arguments: List[str]
    :param start_position: Starting number for the enumeration of arguments, defaults to 0
    :type start_position: int, optional
    :raises (argparse.ArgumentError, argparse.ArgumentTypeError): Argument parsering exceptions
    :return: Three dictionaries with the arguments name as keys and different values: CWL input bidding, CWL type, CWL value
    :rtype: Tuple[MutableMapping[str, Any], MutableMapping[str, str], MutableMapping[str, Any]]
    """
    # Three dictionaries are produced, with three different mappings
    arg_to_bidding, arg_to_type, arg_to_value = {}, {}, {}

    # Parse the command line arguments and convert the namespace to a dictionary
    try:
        namespace = vars(parser.parse_args(arguments))
        print(namespace)
    except (argparse.ArgumentError, argparse.ArgumentTypeError) as e:
        raise e

    # Itarate over the parser's declared arguments and compile the three dictionaries
    for position, action in enumerate(parser._actions, start=start_position):
        if not isinstance(action, _HelpAction):
            # Convert the action attributes into useful parameter informations
            input = get_param_name(action.option_strings, parser.prefix_chars)
            cwl_type = CWL_TYPE_MAPPING[action.type]
            required = action.required

            # Model does not need to be added to the configurations, it is handled separately
            if not input == "model":
                # Argument name to CWL input bidding format
                arg_to_bidding[input] = {
                    "type": cwl_type + ("" if required else "?"),
                    "inputBinding": {
                        "position": position,
                        "prefix": get_param_flag(action.option_strings),
                    },
                    "default": action.default,
                }

                # Argument name to CWL type
                arg_to_type[input] = cwl_type + ("" if required else "?")

                # Argument name to value (Directory and folder require different format)
                namespace_input = (
                    input
                    if input in namespace
                    else action.dest  # Argmuents can be stored in a variable with a name different from their flag, namely dest
                )
                arg_to_value[input] = (
                    namespace[namespace_input]
                    if action.type not in [FolderLike, FileLike]
                    else {
                        "class": cwl_type,
                        "path": namespace[namespace_input],
                    }
                )

    return arg_to_bidding, arg_to_type, arg_to_value


def import_from_path(module_name: str, file_path: FileLike) -> types.ModuleType:
    """Dinamically import a module from a file

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
