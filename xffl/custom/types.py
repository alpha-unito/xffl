"""Custom types for better type hints in xFFL."""

from pathlib import Path
from typing import Union

from xffl.utils.utils import resolve_path


class PathLike(Path):
    """Path-like object (file or folder)."""

    def __init__(self, path: Union[str, Path]) -> None:
        _path: Path = path if isinstance(path, Path) else Path(path)
        if not _path.is_absolute():
            _path = resolve_path(path=_path)
        super().__init__(_path)

    def __repr__(self) -> str:
        return super().__str__()


class FolderLike(PathLike):
    """Path to a folder."""

    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__(path)

        if not self.is_dir():
            raise ValueError(f"Invalid directory path: {self}")


class FileLike(PathLike):
    """Path to a file."""

    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__(path)

        if not self.is_file():
            raise ValueError(f"Invalid file path: {self}")
