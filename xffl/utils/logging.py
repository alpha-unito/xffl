"""Logging utilities for xFFL."""

import logging
import sys

from xffl.custom.formatter import CustomFormatter, ExcludeLoggerFilter

# Logger names to exclude from formatting
EXCLUDED_LOGGERS = ["asyncio", "git"]


def setup_logging(log_level: int = logging.INFO) -> None:
    """
    Set up the global logging configuration and root xFFL logger.

    This function configures a default StreamHandler with the CustomFormatter,
    applies the ExcludeLoggerFilter to ignore unwanted loggers, and sets the
    root logger level.

    :param log_level: Logging level (0-50). Defaults to logging.INFO [20].
    :type log_level: int, optional
    """
    # Clear existing handlers and configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove any pre-existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add default handler
    root_logger.addHandler(get_default_handler(log_level=log_level))


def get_default_handler(log_level: int = logging.INFO) -> logging.StreamHandler:
    """
    Create and return the default xFFL logging StreamHandler.

    The handler uses CustomFormatter for colored output and an
    ExcludeLoggerFilter to ignore unwanted loggers.

    :param log_level: Logging level (0-50). Defaults to logging.INFO [20].
    :type log_level: int, optional
    :return: Configured StreamHandler for xFFL logging
    :rtype: logging.StreamHandler
    """
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(CustomFormatter())
    handler.addFilter(ExcludeLoggerFilter(exclude=EXCLUDED_LOGGERS))

    return handler
