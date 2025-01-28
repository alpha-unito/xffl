"""Data-related utility methods"""

from logging import Logger, getLogger
from typing import Dict, Union

from datasets import Dataset, DatasetDict, load_from_disk

from xffl.custom.types import PathLike

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def load_datasets_from_disk(
    paths: Dict[str, PathLike]
) -> Dict[str, Union[Dataset, DatasetDict]]:
    """Load multiple datasets from disk

    Useful when train, test, and validation sets are different files/folders

    :param paths: Dictionary of "split_name": path_to_the_dataset_split
    :type paths: Dict[str, PathLike]
    :return: Dictionary of "split_name": HuggingFace_dataset object
    :rtype: Dict[str, Union[Dataset, DatasetDict]]
    """
    datasets = {}
    for split, path in paths.items():
        datasets[split] = load_from_disk(path)

    return datasets
