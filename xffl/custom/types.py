"""Custom types for better type hints"""

import os
from os import PathLike as _PathLike
from pathlib import Path
from typing import Any, TypeVar

from xffl.utils.utils import resolve_path

PathLike: TypeVar = TypeVar("PathLike", str, Path, _PathLike)
"""Path-like objects"""


def PathLike(path: Any) -> PathLike:
    """PathLike objects constructor

    :param path: A file system path
    :type path: Any
    :return: Expanded and resolved file system path
    :rtype: PathLike
    """
    return resolve_path(path=path)


FolderLike: TypeVar = TypeVar("FolderLike", str, Path, _PathLike, PathLike)
"""Path to folder objects"""


def FolderLike(path: Any) -> FolderLike:
    """Folder objects constructor

    :param path: A file system path to a folder
    :type path: Any
    :return: Expanded and resolved file system folder path
    :rtype: PathLike
    """
    if os.path.isdir(PathLike(path=path)):
        return path
    else:
        raise ValueError


FileLike: TypeVar = TypeVar("FileLike", str, Path, _PathLike, PathLike)
"""Path to file objects"""


def FileLike(path: Any) -> FileLike:
    """File objects constructor

    :param path: A file system path to a file
    :type path: Any
    :return: Expanded and resolved file system file path
    :rtype: PathLike
    """
    if os.path.isfile(PathLike(path=path)):
        return path
    else:
        raise ValueError
