"""Custom types for better type hints in xFFL. Local paths are automatically resolved and expanded."""

from pathlib import Path
from typing import Union

from xffl.cli.utils import resolve_path


class _PathLike:
    """Path-like object (file or folder)."""

    def __init__(self, path: Union[str, Path], local=True):
        p: Path = Path(path)
        if local and not p.is_absolute():
            p = resolve_path(p)
        self.path = p

    def __str__(self):
        return str(self.path)

    def __fspath__(self):
        return self.path


class FolderLike(_PathLike):
    """Path to a folder."""

    def __init__(self, path: Union[str, Path], local=True):
        super().__init__(path=path, local=local)
        if local and not self.path.is_dir():
            raise ValueError(f"Invalid directory path: {self.path}")


class FileLike(_PathLike):
    """Path to a file."""

    def __init__(self, path: Union[str, Path], local=True):
        super().__init__(path=path, local=local)
        if local and not self.path.is_file():
            raise ValueError(f"Invalid directory path: {self.path}")
