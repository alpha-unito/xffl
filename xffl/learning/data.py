"""Data-related utility methods"""

import os
from logging import Logger, getLogger
from pathlib import Path
from typing import Dict, Mapping

from datasets import Dataset, DatasetDict, load_from_disk

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def load_datasets_from_disk(
    splits: Mapping[str, Path], base_path: Path
) -> Dict[str, Dataset | DatasetDict]:
    """Load multiple datasets from disk

    Useful when train, test, and validation sets are different files/folders

    :param splits: Dictionary of "split_name": path_to_the_dataset_split
    :type splits: Dict[str, PathLike]
    :param base_path: Base path for the dataset folder
    :type base_path: FolderLike
    :return: Dictionary of "split_name": HuggingFace_dataset object
    :rtype: Dict[str, Union[Dataset, DatasetDict]]
    """
    datasets = {}
    for split, path in splits.items():
        datasets[split] = load_from_disk(os.path.join(base_path, path))

    return datasets
