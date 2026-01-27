"""Utility methods"""

from __future__ import annotations

import os
from collections.abc import Callable
from logging import Logger, getLogger
from pathlib import Path
from typing import Optional

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def resolve_path(path: str | Path) -> Path:
    """Check the path is well formatted, otherwise tries to fix it.

    :param path: abbreviated path
    :type path: str or Path
    :return: expanded path
    :rtype: Path
    """
    return Path(os.path.expanduser(os.path.expandvars(path))).absolute().resolve()


def check_input(
    text: str,
    warning_msg: str,
    control: Optional[Callable] = None,
    is_local_path: bool = False,
    optional: bool = False,
) -> str:
    """Receives and checks a user input based on the specified condition

    :param text: Question to be asked to the user
    :type text: str
    :param warning_msg: Error message in case the inserted value does not satisfy the control condition
    :param warning_msg: Error message in case the inserted value does not satisfy the control condition
    :type warning_msg: str
    :param control: Control function to be checked on the inserted value
    :param control: Control function to be checked on the inserted value
    :type control: Callable
    :param is_local_path: If the provided path is a local path, defaults to True
    :type is_local_path: bool
    :param optional: If the user can leave the input blank, defaults to False.
    :type optional: bool, optional
    :return: The validated user input, or None if the input was left blank and optional is True.
    :rtype: str
    """
    is_valid = control or (lambda _: True)
    while True:
        raw_value = input(text).strip()
        if not raw_value:
            if optional:
                logger.warning("No value provided; skipping.")
                return ""
            continue
        processed_value = (
            str(resolve_path(path=raw_value)) if is_local_path else raw_value
        )
        if is_valid(processed_value):
            return processed_value
        logger.warning(warning_msg.format(processed_value))
