"""Custom types for better type hints in xFFL."""

from pathlib import Path
from typing import Any, Union

from xffl.utils.utils import resolve_path


class PathLike(Path):
    """Path-like object (file or folder)."""

    _flavour = Path()._flavour

    def __new__(cls, path: Union[str, Path, Any]) -> "PathLike":
        resolved = resolve_path(path=path)
        return super().__new__(cls, resolved)

    def __call__(self, path: Any) -> "PathLike":
        return PathLike(path)


class FolderLike(PathLike):
    """Path to a folder."""

    _flavour = Path()._flavour

    def __new__(cls, path: Union[str, Path, Any]) -> "FolderLike":
        resolved = resolve_path(path=path)
        if not resolved.is_dir():
            raise ValueError(f"Invalid folder path: {path}")
        return super().__new__(cls, resolved)

    def __call__(self, path: Any) -> "FolderLike":
        return FolderLike(path)


class FileLike(PathLike):
    """Path to a file."""

    _flavour = Path()._flavour

    def __new__(cls, path: Union[str, Path, Any]) -> "FileLike":
        resolved = resolve_path(path=path)
        if not resolved.is_file():
            raise ValueError(f"Invalid file path: {path}")
        return super().__new__(cls, resolved)

    def __call__(self, path: Any) -> "FileLike":
        return FileLike(path)
