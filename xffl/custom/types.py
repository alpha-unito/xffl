"""Custom types for better type hints"""

from os import PathLike as _PathLike
from pathlib import Path
from typing import TypeVar

PathLike = TypeVar("PathLike", str, Path, _PathLike)
"""Path-like objects"""
