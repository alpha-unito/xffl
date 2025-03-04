"""Utility methods exploitable in many different situations"""

import os
from collections.abc import Callable
from logging import Logger, getLogger
from pathlib import Path, PurePath
from typing import List

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def get_param_flag(list_: List[str]) -> str:
    """Gets the full command line parameter flag

    :param list_: List of the parameter's flags
    :type list_: List[str]
    :return: The full parameter flag
    :rtype: str
    """
    return max(list_, key=len)


def get_param_name(list_: List[str], prefix: str = "-") -> str:
    """Returns the command line parameter full name given its flag list

     This method also replaces scores with underscores

    :param list_: List of the parameter's flags
    :type list_: List[str]
    :param prefix: Prefix symbol preceding a flag, defaults to "-"
    :type prefix: str
    :return: Full parameter name
    :rtype: str
    """

    return get_param_flag(list_=list_).lstrip(prefix).replace(prefix, "_")


def resolve_path(path: str, is_local_path: bool = True) -> str:
    """Check the path is well-formed, otherwise tries to fix it.
    Moreover, the path is resolve to the absolute form if it is defined as a local path

    :param path: abbreviated path
    :type path: str
    :param is_local_path:
    :type is_local_path: bool
    :return: expanded path
    :rtype: str
    """
    return str(
        Path(os.path.expanduser(os.path.expandvars(path))).absolute().resolve()
        if is_local_path
        else PurePath(path)
    )


def check_input(
    text: str,
    warning_msg: str,
    control: Callable,
    is_path: bool = False,
    is_local_path=True,
) -> str:
    """Receives and checks a user input based on the specified condition

    :param text: Question to be asked to the user
    :type text: str
    :param warning_msg: Error message in case the inserted value does not satisfy the control condition
    :type warning_msg: str
    :param control: Control function to be checked on the inserted value
    :type control: Callable
    :param is_path: Flag signaling if the expected input is a path, defaults to False
    :type is_path: bool, optional
    :param is_local_path: Flag signaling if what check the existence of the path in local, defaults to False
    :type is_local_path: bool, optional
    :return: The value inserted from the user satisfying the condition
    :rtype: str
    """
    condition = False
    while not condition:
        value = input(text)
        if is_path:
            value = resolve_path(value, is_local_path)
        if not (condition := control(value)):
            logger.warning(warning_msg.format(value))
    return value
