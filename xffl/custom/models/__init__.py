"""List of supported models"""

from typing import Final, Mapping

from .llama import Llama
from .mixtral import Mixtral
from .model_info import ModelInfo

MODELS: Final[Mapping[str, ModelInfo]] = {
    "tiny-random-llama-3": Llama(
        path="/leonardo_scratch/fast/uToID_bench/xffl/models/tiny-random-llama-3"
    ),
    "llama3.1-8b": Llama(
        path="/leonardo_scratch/fast/uToID_bench/xffl/models/llama3.1-8b"
    ),
    "llama3.1-70b": Llama(
        path="/leonardo_scratch/fast/uToID_bench/xffl/models/llama3.1-70b"
    ),
    "mixtral-8x7b-v0.1": Mixtral(
        path="/leonardo_scratch/fast/uToID_bench/xffl/models/mixtral-8x7b-v0.1"
    ),
}
""" Supported models dictionary """
