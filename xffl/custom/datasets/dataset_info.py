"""Abstract dataset class"""

from dataclasses import dataclass
from typing import Mapping

from xffl.custom.types import FolderLike, PathLike


@dataclass
class DatasetInfo:
    """Abstract dataset essential information

    :param splits: Mapping between the split's name and its location inside the dataset's folder
    :type splits: Mapping[str, PathLike]
    :param path: Path to the dataset's folder
    :type path: FolderLike
    """

    splits: Mapping[str, PathLike]
    path: FolderLike
