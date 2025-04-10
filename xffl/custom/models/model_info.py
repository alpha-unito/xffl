""" Abstract model class """

from abc import ABC
from dataclasses import dataclass
from typing import Callable, Type

from xffl.custom.types import FolderLike


@dataclass
class ModelInfo(ABC):
    """Abstract model essential information"""

    model_type: Type
    decoder_layer: Type
    wrapping_policy: Callable
    path: FolderLike