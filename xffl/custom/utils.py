import os
from logging import Logger, getLogger
from pathlib import Path

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def resolve_path(path: str | Path) -> Path:
    """Check the path is well formatted, otherwise tries to fix it.

    :param path: abbreviated path
    :type path: str or Path
    :return: expanded path
    :rtype: Path
    """
    logger.debug(f"Expanding {path} path...")
    return Path(os.path.expanduser(os.path.expandvars(path))).absolute().resolve()
