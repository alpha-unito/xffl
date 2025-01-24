"""Logging utilities
"""

import logging
import sys

from xffl.custom.formatter import CustomFormatter
from xffl.utils.constants import VERSION


def setup_logging(log_level: int = logging.INFO):
    logging.basicConfig(
        level=log_level,
        handlers=[get_default_handler(log_level=log_level)],
        force=True,
    )
    xffl_logger = logging.getLogger("xffl")
    xffl_logger.info(
        f"Cross-Facility Federated Learning (xFFL) - {VERSION} - Starting execution...",
    )

    set_external_loggers()


def get_default_handler(log_level: int = logging.INFO):
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level=log_level)
    handler.setFormatter(CustomFormatter())
    return handler


def set_external_loggers():
    external_loggers = ["cwltool", "salad", "streamflow"]

    for logger_name in external_loggers:
        logging.getLogger(logger_name).handlers = []
