""" clean_mc4_it modelling """

from dataclasses import dataclass, field
from typing import Mapping

from xffl.custom.types import FolderLike


@dataclass
class clean_mc4_it:
    """clean_mc4_it essential information"""

    splits: Mapping[str, FolderLike] = field(
        default_factory=lambda: {
            "train": "train",
            "val": "val",
        }
    )
    path: FolderLike = "/path/to/llama/model"
