"""Data-related utility methods"""

import os
from logging import Logger, getLogger
from typing import Dict, Optional, Union

from datasets import Dataset, DatasetDict, load_from_disk

from xffl.custom.types import FolderLike, PathLike

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def load_datasets_from_disk(
    splits: Dict[str, PathLike], base_path: Optional[FolderLike] = ""
) -> Dict[str, Union[Dataset, DatasetDict]]:
    """Load multiple datasets from disk

    Useful when train, test, and validation sets are different files/folders

    :param splits: Dictionary of "split_name": path_to_the_dataset_split
    :type splits: Dict[str, PathLike]
    :return: Dictionary of "split_name": HuggingFace_dataset object
    :rtype: Dict[str, Union[Dataset, DatasetDict]]
    """
    datasets = {}
    for split, path in splits.items():
        datasets[split] = load_from_disk(os.path.join(base_path, path))

    return datasets
