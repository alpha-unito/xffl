"""Logging utilities
"""

import logging
import logging.config

from xffl.utils.constants import LOGGING_CONFIGURATION, VERSION

LOGGER_NUMBER: int = 0
"""Number of different loggers currently instantiated"""


def get_logger(
    class_name: str = "root", log_level: int = logging.INFO
) -> logging.Logger:
    """Returns the logger associated to the specified class name.
        All the configuration for the loggers is contained in the logging.conf file.
        A different logger is defined for each class, defined by the class own name.
        This way multiple instances of the same class will obtain the same instance of the class logger.
        when a new class is created, a new entry should be inserted in the logging.conf file.

    :param class_name: a string reporting the class name asking for a logger.
    :type class_name: str

    :return: Logger object.
    :rtype: logging.Logger
    """
    global LOGGER_NUMBER
    LOGGER_NUMBER += 1

    logging.config.fileConfig(LOGGING_CONFIGURATION)
    logging.basicConfig(level=log_level)

    if LOGGER_NUMBER == 1:
        logger = logging.getLogger("root")
        logger.info(
            f"Cross-Facility Federated Learning (xFFL) {VERSION} - %s - Starting the execution...",
        )
    return logging.getLogger(class_name)


def set_log_level(log_level: int = logging.INFO):
    logging.basicConfig(level=log_level)
