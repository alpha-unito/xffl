"""All the constants used in xFFL should be declared and sourced from here"""

import os
from typing import Final

# Version #
VERSION: Final[str] = "v0.2.1"
"""xFFL software version"""

# Paths #
XFFL_CACHE_DIR = os.path.join(
    os.environ.get("XFFL_CACHE_DIR", os.path.expanduser("~")), ".xffl", VERSION
)
