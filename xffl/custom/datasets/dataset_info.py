"""Abstract dataset class"""

from dataclasses import dataclass
from typing import Mapping

from xffl.custom.types import FolderLike, PathLike


@dataclass
class DatasetInfo:
    """Abstract dataset essential information"""

    splits: Mapping[str, PathLike]
    path: FolderLike
