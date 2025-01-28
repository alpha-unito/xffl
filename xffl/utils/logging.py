"""Logging utilities"""

import logging
import sys

from xffl.custom.formatter import CustomFormatter
from xffl.utils.constants import VERSION


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
    xffl_logger = logging.getLogger("xffl")
    xffl_logger.info(
        f"Cross-Facility Federated Learning (xFFL) - {VERSION} - Starting execution...",
    )

    set_external_loggers(log_level=log_level)


def get_default_handler(log_level: int = logging.INFO) -> logging.StreamHandler:
    """Returns xFFL's default logging handler configuration

    :param log_level: log level to be used [0-50], defaults to logging.INFO [20]
    :type log_level: int, optional
    :return: xFFL's default logging handler
    :rtype: logging.StreamHandler
    """
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level=log_level)
    handler.setFormatter(CustomFormatter())
    return handler


def set_external_loggers(log_level: int = logging.INFO):
    """Empties the formatter list of the imported libraries loggers to obtain homogeneous output"""

    if "transformers" in sys.modules:
        import transformers

        transformers.utils.logging._get_library_root_logger().handlers = []
        transformers.utils.logging._get_library_root_logger().propagate = True
        # transformers.utils.logging.set_verbosity(log_level)
        # transformers.utils.logging.disable_default_handler()
        # transformers.utils.logging.disable_progress_bar()
        #    get_default_handler(log_level)
        # ]

    if "datasets" in sys.modules:
        import datasets

        datasets.utils.logging._get_library_root_logger().handlers = []
        datasets.utils.logging._get_library_root_logger().propagate = True
        # datasets.utils.logging.set_verbosity(log_level)
        # datasets.utils.logging.disable_default_handler()
        # datasets.utils.logging.disable_progress_bar()
        # datasets.utils.logging._get_library_root_logger().handlers = [
        #    get_default_handler(log_level)
        # ]

    external_loggers = ["cwltool", "salad", "streamflow"]
    for logger_name in external_loggers:
        logging.getLogger(logger_name).handlers = []
