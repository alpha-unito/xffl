"""Abstract model class"""

from abc import ABC
from dataclasses import dataclass
from typing import Callable, Type

from xffl.custom.types import FolderLike


@dataclass
class ModelInfo(ABC):
    """Abstract model essential information

    :param model_type: Class of the selected model
    :type model_type: Type
    :param decoder_layer: Class of the decoder layer of the model
    :type decoder_layer: Type
    :param wrapping_policy: FSDP wrapping function for the decoder layer
    :type wrapping_policy: Callable
    :param path: Path to the model's folder
    :type path: FolderLike
    """

    model_type: Type
    decoder_layer: Type
    wrapping_policy: Callable
    path: FolderLike
