"""Utility methods for the workflow configuration creation"""

import argparse
from argparse import _HelpAction
from types import MappingProxyType
from typing import Any, Final, List, MutableMapping

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

    return get_param_flag(list=list).replace(prefix * 2, "").replace("-", "_")


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


def from_parser_to_cwl(
    parser: argparse.ArgumentParser, arguments=List[str], start_position: int = 0
) -> MutableMapping[str, Any]:
    """Method that converts a Python ArgumentParser into a valid dictionary of CWL inputs entries

    :param parser: Python command line argument parser
    :type parser: argparse.ArgumentParser
    :return: Dictionary of valid CWL inputs entries
    :rtype: MutableMapping[str, MutableMapping[str, str | MutableMapping[str, str]]]
    """
    training_step_args, main_cwl_args, round_cwl_args, config_cwl_args = {}, {}, {}, {}

    namespace = vars(parser.parse_args(arguments))

    for position, action in enumerate(parser._actions, start=start_position):
        if not isinstance(action, _HelpAction):
            param_name = get_param_name(action.option_strings, parser.prefix_chars)

            training_step_args[param_name] = {
                "type": CWL_TYPE_MAPPING[action.type]
                + ("" if action.required else "?"),
                "inputBinding": {
                    "position": position,
                    "prefix": get_param_flag(action.option_strings),
                },
                "default": action.default if action.default is not None else "null",
            }

            main_cwl_args[param_name] = CWL_TYPE_MAPPING[action.type] + (
                "" if action.required else "?"
            )

            config_cwl_args[param_name] = (
                namespace[param_name]
                if action.type not in [FolderLike, FileLike]
                else {
                    "class": CWL_TYPE_MAPPING[action.type],
                    "path": namespace[param_name],
                }
            )

    round_cwl_args = main_cwl_args

    return training_step_args, main_cwl_args, round_cwl_args, config_cwl_args
