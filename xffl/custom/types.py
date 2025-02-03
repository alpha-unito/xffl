"""Custom types for better type hints"""

from os import PathLike as _PathLike
from pathlib import Path
from typing import Any, TypeVar

from xffl.utils.utils import resolve_path

PathLike = TypeVar("PathLike", str, Path, _PathLike)
"""Path-like objects"""


def PathLike(path: Any) -> PathLike:
    """PathLike objects constructor

    :param path: A file system path
    :type path: Any
    :return: Expanded and resolved file system path
    :rtype: PathLike
    """
    return resolve_path(path=path)
