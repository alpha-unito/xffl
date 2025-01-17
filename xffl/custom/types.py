"""Custom types for better type hints"""

from os import PathLike as _PathLike
from pathlib import Path
from typing import Final, TypeVar

PathLike: Final[TypeVar] = TypeVar("PathLike", str, Path, _PathLike)
"""Path-like objects"""
