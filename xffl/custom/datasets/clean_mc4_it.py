"""clean_mc4_it modelling"""

from dataclasses import dataclass, field
from typing import Mapping

from xffl.custom.types import FolderLike, PathLike

from .dataset_info import DatasetInfo


@dataclass
class CleanMc4It(DatasetInfo):
    """clean_mc4_it essential information"""

    splits: Mapping[str, PathLike] = field(
        default_factory=lambda: {
            "train": "train",
            "val": "val",
        }
    )
    path: FolderLike = "/path/to/clean_mc4_it/dataset"
