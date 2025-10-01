"""List of supported datasets"""

from typing import Final, Mapping

from .clean_mc4_it import CleanMc4It
from .dataset_info import DatasetInfo

DATASETS: Final[Mapping[str, DatasetInfo]] = {
    "clean_mc4_it": CleanMc4It(
        path="/leonardo_scratch/fast/uToID_bench/xffl/data/clean_mc4_it"
    ),
}
""" Supported datasets dictionary """
