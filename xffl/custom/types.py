"""Custom types for better type hints in xFFL."""

from pathlib import Path
from typing import Any, NewType

from xffl.utils.utils import resolve_path

# --- Type aliases ---
PathLike = NewType("PathLike", Path)
"""Path-like object (file or folder)."""

FolderLike = NewType("FolderLike", Path)
"""Path to a folder."""

FileLike = NewType("FileLike", Path)
"""Path to a file."""


# --- Constructors ---
def as_pathlike(path: Any) -> PathLike:
    """Convert input into a resolved PathLike.

    :param path: File system path (string, Path, etc.)
    :type path: Any
    :return: Resolved filesystem path
    :rtype: PathLike
    """
    return PathLike(resolve_path(path=path))


def as_folderlike(path: Any) -> FolderLike:
    """Convert input into a resolved FolderLike.

    :param path: Path to a folder
    :type path: Any
    :raises ValueError: If the path is not a valid directory
    :return: Resolved folder path
    :rtype: FolderLike
    """
    resolved = Path(resolve_path(path=path))
    if resolved.is_dir():
        return FolderLike(resolved)
    raise ValueError(f"Invalid folder path: {path}")


def as_filelike(path: Any) -> FileLike:
    """Convert input into a resolved FileLike.

    :param path: Path to a file
    :type path: Any
    :raises ValueError: If the path is not a valid file
    :return: Resolved file path
    :rtype: FileLike
    """
    resolved = Path(resolve_path(path=path))
    if resolved.is_file():
        return FileLike(resolved)
    raise ValueError(f"Invalid file path: {path}")
