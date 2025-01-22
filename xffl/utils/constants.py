"""All the constants used in xFFL should be declared and sourced from here
"""

import os
from typing import Final

import xffl
from xffl.custom.types import PathLike

# Version #
VERSION: Final[str] = "v0.1.0"
"""xFFL software version"""

# Paths #
DEFAULT_xFFL_DIR: Final[PathLike] = os.path.dirname(os.path.abspath(xffl.__file__))  # type: ignore
"""FastFederatedLearning deault  root directory"""

# Logging #
LOGGING_CONFIGURATION: Final[PathLike] = os.path.join(DEFAULT_xFFL_DIR, "utils/logging.conf")  # type: ignore

"""Path to the logging configuration file"""
