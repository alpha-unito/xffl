"""Custom logging.Formatter for formatted and coloured logging."""

import logging
from logging import LogRecord
from typing import Literal, Optional


class CustomFormatter(logging.Formatter):
    """Logging formatter with color support per log level.

    Colors adapted from https://stackoverflow.com/a/56944256/3638629.
    """

    # ANSI color codes
    GREY = "\x1b[38;20m"  # Info
    BLUE = "\x1b[38;5;39m"  # Debug
    YELLOW = "\x1b[38;5;226m"  # Warning
    RED = "\x1b[38;5;196m"  # Error
    BOLD_RED = "\x1b[31;1m"  # Critical
    RESET = "\x1b[0m"

    DEFAULT_FORMAT = "%(asctime)s | %(name)16s | %(levelname)8s | %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"  # Less detailed: only hours:minutes:seconds

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: Literal["%", "{", "$"] = "%",
        validate: bool = True,
    ) -> None:
        """Initialize the CustomFormatter.

        :param fmt: Format string for log messages. Defaults to a structured format.
        :type fmt: Optional[str]
        :param datefmt: Date format string.
        :type datefmt: Optional[str]
        :param style: Style of the format string ("%", "{", or "$").
        :type style: Literal["%", "{", "$"]
        :param validate: Whether to validate the format string.
        :type validate: bool
        """
        super().__init__(
            fmt=fmt or self.DEFAULT_FORMAT,
            datefmt=datefmt or self.DATE_FORMAT,
            style=style,
            validate=validate,
        )

        self.fmt = fmt or self.DEFAULT_FORMAT

        # Mapping log levels to colored formats
        self.FORMATS = {
            logging.DEBUG: self.BLUE + self.fmt + self.RESET,
            logging.INFO: self.GREY + self.fmt + self.RESET,
            logging.WARNING: self.YELLOW + self.fmt + self.RESET,
            logging.ERROR: self.RED + self.fmt + self.RESET,
            logging.CRITICAL: self.BOLD_RED + self.fmt + self.RESET,
        }

    def format(self, record: LogRecord) -> str:
        """Format the provided log record with level-specific color.

        :param record: The log record to format.
        :type record: LogRecord
        :return: The formatted log message as a string.
        :rtype: str
        """
        formatter = logging.Formatter(
            self.FORMATS.get(record.levelno, self.fmt), datefmt=self.DATE_FORMAT
        )
        return formatter.format(record)
