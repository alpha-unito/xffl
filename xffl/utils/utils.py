"""Utility methods exploitable in many different situations"""

from __future__ import annotations

from datetime import timedelta
from logging import Logger, getLogger
from typing import Any, Optional

from torch.distributed import ProcessGroupNCCL

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


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
    options: ProcessGroupNCCL.Options = ProcessGroupNCCL.Options()
    options.is_high_priority_stream = is_high_priority_stream
    options._timeout = get_timeout()

    return options


def resolve_param(
    value: Any,
    config: Any,
    attr: str,
) -> Optional[Any]:
    """Resolve a function parameter giving priority to the "value" field, alternatively tries to get it from the xFFL configuration.
       If both fail, None is returned.

    :param value: Given parameter value
    :type value: Optional[Any]
    :param config: xFFL configuration
    :type config: Optional[XFFLConfig]
    :param attr: Name of the attribute in the xFFL configuration
    :type attr: str
    :return: Resolved value of the parameter, None if none is found
    :rtype: Optional[Any]
    """
    return value if value is not None else getattr(config, attr, None)
