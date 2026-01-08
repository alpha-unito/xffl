"""All the constants used in xFFL should be declared and sourced from here"""

import os
from pathlib import Path
from typing import Final

import xffl

# Version #
VERSION: Final[str] = "v0.2.1"
"""xFFL software version"""

# Paths #
DEFAULT_xFFL_DIR: Final[Path] = Path(os.path.dirname(os.path.abspath(xffl.__file__)))
"""FastFederatedLearning default root directory"""

SUPPORTED_QUEUE_MANAGERS: Final[list] = [
    "flux",
    "slurm",
    "pbs",  # "lsf" installing streamflow-lsf plugin
]
