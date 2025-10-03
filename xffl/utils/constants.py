"""All the constants used in xFFL should be declared and sourced from here"""

import os
from typing import Final

import xffl
from xffl.custom.types import PathLike

# Version #
VERSION: Final[str] = "v0.2.1"
"""xFFL software version"""

# Paths #
DEFAULT_xFFL_DIR: Final[PathLike] = os.path.dirname(os.path.abspath(xffl.__file__))
"""FastFederatedLearning default root directory"""
