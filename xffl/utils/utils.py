"""Utility methods exploitable in many different situations
"""

import os
from collections.abc import Callable
from logging import Logger, getLogger
from pathlib import Path

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def resolve_path(path: str) -> str:
    """Converts a relative, shortened path into an absolute one

    :param path: abbreviated path
    :type path: str
    :return: expanded path
    :rtype: str
    """

    return str(Path(os.path.expanduser(os.path.expandvars(path))).absolute())


def check_input(
    text: str, warning_msg: str, control: Callable, is_path: bool = False
) -> str:
    """Receives and checks a user input based on the specified condition

    :param text: Question to be asked to the user
    :type text: str
    :param warning_msg: Error message in case the inserted value does not satisfies the control condition
    :type warning_msg: str
    :param control: Control function to be cheked on the inserted value
    :type control: Callable
    :param is_path: Flag signaling if the expected input is a path, defaults to False
    :type is_path: bool, optional
    :return: The value inserted from the user satisfying the condition
    :rtype: str
    """

    condition = False
    while not condition:
        value = input(text)
        if is_path:
            value = resolve_path(value)
        if not (condition := control(value)):
            logger.warning(warning_msg.format(value))
    return value
