""" List of already supported models and datasets"""

from typing import Callable, Final, Mapping

from .datasets.clean_mc4_it import clean_mc4_it
from .models.llama import llama
from .models.mixtral import mixtral

MODELS: Final[Mapping[str, Callable]] = {
    "llama3.1-8b": llama,
    "llama3.1-70b": llama,
    "mixtral-8x7b": mixtral,
}
""" Supported models dictionary """

DATASETS: Final[Mapping[str, Callable]] = {
    "clean_mc4_it": clean_mc4_it,
}
""" Supported datasets dictionary """
