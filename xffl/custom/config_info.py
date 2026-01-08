"""Abstract dataset class"""

from abc import ABC
from dataclasses import dataclass
from typing import Callable, Mapping, Type


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
    :type path: str
    """

    model_type: str
    decoder_layers: Type
    wrapping_policy: Callable
    name: str
    path: str = "/model/"


@dataclass
class DatasetInfo(ABC):
    """Abstract dataset essential information

    :param splits: Mapping between the split's name and its location inside the dataset's folder
    :type splits: Mapping[str, str]
    :param path: Path to the dataset's folder
    :type path: str
    """

    name: str
    splits: Mapping[str, str]
    path: str = "/dataset/"


@dataclass
class XFFLConfig(ABC):
    """Abstract base XFFL configuration

    :param model: Model configuration
    :type model: ModelInfo
    :param dataset: Dataset configuration
    :type dataset: DatasetInfo
    """

    model: ModelInfo
    dataset: DatasetInfo
