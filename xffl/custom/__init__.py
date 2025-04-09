"""List of already supported models and datasets"""

from typing import Callable, Final, Mapping

from .datasets.clean_mc4_it import CleanMc4It
from .datasets.dataset_info import DatasetInfo
from .models.llama import Llama
from .models.mixtral import Mixtral
from .models.model_info import ModelInfo

MODELS: Final[Mapping[str, Callable]] = {
    "llama3.1-8b": Llama,
    "llama3.1-70b": Llama,
    "mixtral-8x7b": Mixtral,
}
""" Supported models dictionary """

DATASETS: Final[Mapping[str, Callable]] = {
    "clean_mc4_it": CleanMc4It,
}
""" Supported datasets dictionary """
