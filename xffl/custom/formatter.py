"""Custom logging.Formatter for formatted and coloured logging with logger filtering."""

import logging
from logging import LogRecord
from typing import List, Literal, Optional


class ExcludeLoggerFilter(logging.Filter):
    """Filter to exclude logs from specified loggers."""

    def __init__(self, exclude: Optional[List[str]] = None) -> None:
        """
        :param exclude: List of logger names or prefixes to exclude.
        """
        super().__init__()
        self.exclude = set(exclude or [])

    def filter(self, record: LogRecord) -> bool:
        """Return True if the log should be emitted, False otherwise."""
        return not any(record.name.startswith(name) for name in self.exclude)


class CustomFormatter(logging.Formatter):
    """Logging formatter with color support per log level."""

    # ANSI color codes
    GREY = "\x1b[38;20m"  # Info
    BLUE = "\x1b[38;5;39m"  # Debug
    YELLOW = "\x1b[38;5;226m"  # Warning
    RED = "\x1b[38;5;196m"  # Error
    BOLD_RED = "\x1b[31;1m"  # Critical
    RESET = "\x1b[0m"

    DEFAULT_FORMAT = "%(asctime)s | %(name)16s | %(levelname)8s | %(message)s"
    DATE_FORMAT = "%H:%M:%S"

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: Literal["%", "{", "$"] = "%",
        validate: bool = True,
    ) -> None:
        """
        Initialize the CustomFormatter.

        :param fmt: Format string for log messages.
        :param datefmt: Date format string.
        :param style: Style of the format string ("%", "{", or "$").
        :param validate: Whether to validate the format string.
        """
        super().__init__(
            fmt=fmt or self.DEFAULT_FORMAT,
            datefmt=datefmt or self.DATE_FORMAT,
            style=style,
            validate=validate,
        )
        self.fmt = fmt or self.DEFAULT_FORMAT

        self.FORMATS = {
            logging.DEBUG: self.BLUE + self.fmt + self.RESET,
            logging.INFO: self.GREY + self.fmt + self.RESET,
            logging.WARNING: self.YELLOW + self.fmt + self.RESET,
            logging.ERROR: self.RED + self.fmt + self.RESET,
            logging.CRITICAL: self.BOLD_RED + self.fmt + self.RESET,
        }

    def format(self, record: LogRecord) -> str:
        """Format the log record with level-specific color."""
        formatter = logging.Formatter(
            self.FORMATS.get(record.levelno, self.fmt), datefmt=self.DATE_FORMAT
        )
        return formatter.format(record)
