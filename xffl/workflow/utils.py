"""Utility methods for the workflow configuration creation"""

import argparse
from argparse import _HelpAction
from types import MappingProxyType
from typing import Any, Final, List, MutableMapping

from xffl.custom.types import File, Folder


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
    return get_param_flag(list=list).replace(prefix, "").replace("-", "_")


CWL_TYPE_MAPPING: Final[MappingProxyType[Any, str]] = MappingProxyType(
    {
        str: "string",
        int: "int",
        float: "float",
        bool: "boolean",
        None: "boolean",
        Folder: "Folder",
        File: "File",
    }
)
"""An immutable dictionary mapping Python to CWL types"""


def from_parser_to_cwl(
    parser: argparse.ArgumentParser,
) -> MutableMapping[str, MutableMapping[str, str | MutableMapping[str, str]]]:
    """Method that converts a Python ArgumentParser into a valid dictionary of CWL inputs entries

    :param parser: Python command line argument parser
    :type parser: argparse.ArgumentParser
    :return: Dictionary of valid CWL inputs entries
    :rtype: MutableMapping[str, MutableMapping[str, str | MutableMapping[str, str]]]
    """
    inputs = {}
    for position, action in enumerate(parser._actions):
        if not isinstance(action, _HelpAction):
            inputs[get_param_name(action.option_strings, parser.prefix_chars)] = {
                "type": CWL_TYPE_MAPPING[action.type]
                + ("" if action.required else "?"),
                "inputBinding": {
                    "position": position,
                    "prefix": get_param_flag(action.option_strings),
                },
                "default": action.default,
            }

    return inputs
