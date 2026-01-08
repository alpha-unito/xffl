"""Utility methods exploitable in many different situations"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import timedelta
from logging import Logger, getLogger
from pathlib import Path
from typing import Optional, Sequence

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def get_param_flag(flag_list: Sequence[str]) -> str:
    """Gets the full command line parameter flag

    :param flag_list: List of the parameter's flags
    :type flag_list: List[str]
    :return: The full parameter flag
    :rtype: str
    """
    return max(flag_list, key=len)


def get_param_name(flag_list: Sequence[str], prefix: str = "-") -> str:
    """Returns the command line parameter full name given its flag list

     This method also replaces scores with underscores

    :param flag_list: List of the parameter's flags
    :type flag_list: List[str]
    :param prefix: Prefix symbol preceding a flag, defaults to "-"
    :type prefix: str
    :return: Full parameter name
    :rtype: str
    """
    return get_param_flag(flag_list=flag_list).lstrip(prefix).replace(prefix, "_")


def resolve_path(path: str | Path) -> Path:
    """Check the path is well formatted, otherwise tries to fix it.

    :param path: abbreviated path
    :type path: str or Path
    :return: expanded path
    :rtype: Path
    """
    logger.debug(f"Expanding {path} path...")
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
    :type warning_msg: str
    :param control: Control function to be checked on the inserted value
    :type control: Callable
    :param is_local_path: If the provided path is a local path, defaults to True
    :type is_local_path: bool
    :param optional: If True, the user can leave the input blank, returning None. Defaults to False.
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


def get_timeout(
    seconds: float = 120.0,
) -> timedelta:
    """Maximum allowed timeout for distributed communications

    :param seconds: Maximum allowed timeout in seconds, defaults to 12
    :type seconds: Optional[int], optional
    :return: Maximum allowed time delta
    :rtype: timedelta
    """
    return timedelta(seconds=seconds)


def get_default_nccl_process_group_options(
    is_high_priority_stream: bool = True,
):
    """Default NCCL backend configuration for xFFL

    :param is_high_priority_stream: Whether to pick up the highest priority CUDA stream, defaults to True
    :type is_high_priority_stream: Optional[bool], optional
    :return: Configured options for the NCCL backend
    :rtype: ProcessGroupNCCL.Options
    """
    from torch.distributed import ProcessGroupNCCL

    options: ProcessGroupNCCL.Options = ProcessGroupNCCL.Options()
    options.is_high_priority_stream = is_high_priority_stream
    options._timeout = get_timeout()

    return options
