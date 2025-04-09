"""Logging utilities"""

import logging
import sys

from xffl.custom.formatter import CustomFormatter


def setup_logging(log_level: int = logging.INFO):
    """Logging infrastructure setup

    Sets up the global basic logging configuration and the root xffl logger

    :param log_level: log level to be used [0-50], defaults to logging.INFO [20]
    :type log_level: int, optional
    """
    logging.basicConfig(
        level=log_level,
        handlers=[get_default_handler(log_level=log_level)],
        force=True,
    )

    # set_external_loggers()


def get_default_handler(log_level: int = logging.INFO) -> logging.StreamHandler:
    """Returns the xFFL default logging handler configuration

    :param log_level: log level to be used [0-50], defaults to logging.INFO [20]
    :type log_level: int, optional
    :return: xFFL default logging handler
    :rtype: logging.StreamHandler
    """
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level=log_level)
    handler.setFormatter(CustomFormatter())
    return handler


def set_external_loggers():
    """Empties the formatter list of the imported libraries loggers to obtain homogeneous output"""

    if "transformers" in sys.modules:
        import transformers

        transformers.utils.logging._get_library_root_logger().handlers = []
        transformers.utils.logging._get_library_root_logger().propagate = True

    if "datasets" in sys.modules:
        import datasets

        datasets.utils.logging._get_library_root_logger().handlers = []
        datasets.utils.logging._get_library_root_logger().propagate = True

    for logger_name in logging.root.manager.loggerDict:
        if not (logger_name.startswith("xffl") or logger_name.startswith("streamflow")):
            curr_logger = logging.getLogger(logger_name)
            curr_logger.handlers.clear()
            curr_logger.propagate = False
