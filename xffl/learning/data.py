"""Data-related utility methods"""

import os
import random
from logging import Logger, getLogger
from pathlib import Path
from typing import Callable, Dict, Mapping, MutableMapping, Optional, Sequence

import torch
from datasets import DatasetDict, load_from_disk
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from xffl.custom.config import DatasetInfo, XFFLConfig
from xffl.distributed.distributed import DistributedState
from xffl.learning import utils
from xffl.utils.utils import resolve_param

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""

# --------------------------------------------------------------------------- #
#                             Helper methods                                  #
# --------------------------------------------------------------------------- #


def _apply_filters(
    key: str,
    split: Dataset,
    filters: (
        Callable
        | Sequence[Callable]
        | Mapping[str, Callable]
        | Mapping[str, Sequence[Callable]]
    ),
    state: DistributedState,
    config: Optional[XFFLConfig],
) -> Dataset:
    """Applies filter function to all splits of a dataset.

    :param key: Current split name
    :type key: str
    :param split: Current split dataset
    :type split: Dataset
    :param filters: Functions to be applied to the dataset splits before instantiating the dataloaders
    :type filters: Callable | Sequence[Callable] | Mapping[str, Callable] | Mapping[str, Sequence[Callable]]
    :param state: xFFL distributed state
    :type state: DistributedState
    :param config: xFFL configuration, defaults to None
    :type config: Optional[XFFLConfig], optional
    """
    _filters: Callable | Sequence[Callable]
    if isinstance(filters, Mapping):
        _filters = filters[key]
    else:
        _filters = filters

    __filters: Sequence[Callable]
    if isinstance(_filters, Callable):
        __filters = (_filters,)
    else:
        __filters = _filters

    for filter in __filters:
        split = filter(dataset=split, config=config, state=state)

    return split


def _apply_subsampling(
    key: str,
    split: Dataset,
    subsampling: int | Mapping[str, int],
) -> Dataset:
    """Applies subsampling to all splits of a dataset.

    :param key: Current split name
    :type key: str
    :param split: Current split dataset
    :type split: Dataset
    :param subsampling: Number of samples to extract from the dataset splits, defaults to None
    :type subsampling: int | Mapping[str, int]
    """
    _subsampling: int
    if isinstance(subsampling, Mapping):
        _subsampling = subsampling[key]
    else:
        _subsampling = subsampling

    return Subset(split, range(_subsampling))


# --------------------------------------------------------------------------- #
#                             Main Methods.                                   #
# --------------------------------------------------------------------------- #


def load_datasets_from_disk(
    splits: Mapping[str, str], base_path: Path
) -> Dict[str, Dataset | DatasetDict]:
    """Load multiple datasets from disk

    Useful when train, test, and validation sets are different files/folders

    :param splits: Dictionary of "split_name": path_to_the_dataset_split
    :type splits: Dict[str, Path]
    :param base_path: Base path for the dataset folder
    :type base_path: FolderLike
    :return: Dictionary of "split_name": HuggingFace_dataset object
    :rtype: Dict[str, Union[Dataset, DatasetDict]]
    """
    datasets = {}
    for split, path in splits.items():
        datasets[split] = load_from_disk(os.path.join(base_path, path))

    return datasets


def create_dataloaders(
    state: DistributedState,
    dataset: Optional[MutableMapping[str, Dataset]] = None,
    filters: Optional[
        Callable
        | Sequence[Callable]
        | Mapping[str, Callable]
        | Mapping[str, Sequence[Callable]]
    ] = None,
    subsampling: Optional[int | Mapping[str, int]] = None,
    batch_sizes: Optional[Mapping[str, int]] = None,
    workers: Optional[int] = None,
    config: Optional[XFFLConfig] = None,
    collate_fn: Optional[Callable | Mapping[str, Callable]] = None,
    shuffle_train_split: bool = True,
    distributed_sampling: bool = True,
    generator: Optional[torch.Generator] = None,
) -> MutableMapping[str, DataLoader]:
    """Create PyTorch dataloaders based on the provided xFFL configuration or parameters.

    The parameters can be provided both directly and through an XFFL configuration.
    In case both are provided, the firsts take the precedence.

    :param state: xFFL distributed state
    :type state: DistributedState
    :param dataset: Dictionary associating split name with the relative dataset instances, defaults to None
    :type dataset: Optional[MutableMapping[str, Dataset]], optional
    :param filters: Functions to be applied to the dataset splits before instantiating the dataloaders, defaults to None
    :type filters: Optional[Callable | Sequence[Callable] | Mapping[str, Callable] | Mapping[str, Sequence[Callable]]], optional
    :param subsampling: Number of samples to extract from the dataset splits, defaults to None
    :type subsampling: Optional[int | Mapping[str, int]], optional
    :param batch_sizes: Batch size associated to the given dataset splits, defaults to None
    :type batch_sizes: Optional[Mapping[str, int]], optional
    :param workers: Dataloader worker to be instantiated, defaults to None
    :type workers: Optional[int], optional
    :param config: xFFL configuration, defaults to None
    :type config: Optional[XFFLConfig], optionalù
    :param collate_fn: Collate function to apply to the dataloaders, defaults to None
    :type collate_fn: Optional[Callable | Mapping[str, Callable]], optional
    :param shuffle_train_split: If to shuffle the "train" split, defaults to True
    :type shuffle_train_split: bool, optional
    :param distributed_sampling: If to instantiate a DistributedSampler, required if all processes are fed from the same data pool, defaults to True
    :type distributed_sampling: bool, optional
    :param generator: PyTorch RNG state, required for reproducibility, defaults to None
    :type generator: Optional[torch.Generator], optional
    :raises ValueError: If no valid dataset is provided
    :return: Dictionary associating dataset split name and relative dataloader
    :rtype: MutableMapping[str, DataLoader]
    """

    # Parameters resolution
    if config is not None:
        dataset_info: DatasetInfo = config.dataset_info

        if dataset is None:
            __dataset: Optional[Callable] = resolve_param(
                value=dataset, config=dataset_info, attr="splits"
            )
            if __dataset is not None:
                _dataset: Optional[MutableMapping[str, Dataset]] = __dataset(
                    config=config, state=state
                )
        _filters: Optional[
            Callable
            | Sequence[Callable]
            | Mapping[str, Callable]
            | Mapping[str, Sequence[Callable]]
        ] = resolve_param(value=filters, config=dataset_info, attr="filters")
        _collate_fn: Optional[Callable | Mapping[str, Callable]] = resolve_param(
            value=collate_fn, config=dataset_info, attr="collate_fn"
        )
        _subsampling: Optional[int | Mapping[str, int]] = resolve_param(
            value=subsampling, config=dataset_info, attr="subsampling"
        )
        _batch_sizes: Optional[Mapping[str, int]] = resolve_param(
            value=batch_sizes, config=dataset_info, attr="batch_sizes"
        )
        _workers: Optional[int] = resolve_param(
            value=workers, config=dataset_info, attr="workers"
        )
    else:
        _dataset: Optional[MutableMapping[str, Dataset]] = dataset
        _filters: Optional[
            Callable
            | Sequence[Callable]
            | Mapping[str, Callable]
            | Mapping[str, Sequence[Callable]]
        ] = filters
        _collate_fn: Optional[Callable | Mapping[str, Callable]] = collate_fn
        _subsampling: Optional[int | Mapping[str, int]] = subsampling
        _batch_sizes: Optional[Mapping[str, int]] = batch_sizes
        _workers: Optional[int] = workers

    if _dataset is None:
        logger.critical("Impossible setting up the dataloaders: no dataset provided.")
        raise ValueError("Impossible setting up the dataloaders: no dataset provided.")

    # Dataloaders creation
    dataloaders: MutableMapping[str, DataLoader] = {}

    for key, split in _dataset.items():

        # Filters application
        if _filters is not None:
            split = _apply_filters(
                key=key, split=split, filters=_filters, state=state, config=config
            )

        # Subsampling
        if _subsampling is not None:
            split = _apply_subsampling(key=key, split=split, subsampling=_subsampling)

        _batch_size: int = 1
        if _batch_sizes is not None:
            if key in _batch_sizes.keys():
                _batch_size = _batch_sizes[key]
            else:
                logger.error(
                    f"No batch size associated to the {key} split found - defaulting to 1."
                )
        else:
            logger.error(
                "No batch size dictionary found - defaulting all batch sizes to 1."
            )

        __collate_fn: Optional[Callable | Mapping[str, Callable]] = None
        if _collate_fn is not None:
            if isinstance(_collate_fn, Callable):
                __collate_fn = _collate_fn
            else:
                __collate_fn = _collate_fn[key]

        dataloaders[key] = DataLoader(
            dataset=split,
            batch_size=_batch_size,
            sampler=(
                DistributedSampler(
                    dataset=split,
                    num_replicas=state.world_size,
                    rank=state.rank,
                    shuffle=(shuffle_train_split and key == "train"),
                    seed=(
                        config.seed
                        if config is not None and config.seed is not None
                        else random.randint(0, 100)
                    ),
                    drop_last=True,
                )
                if distributed_sampling
                else None
            ),
            num_workers=_workers if _workers else 0,
            collate_fn=__collate_fn,
            pin_memory=True if state.device_type == "cuda" else False,
            drop_last=True,
            worker_init_fn=(
                utils.seed_dataloader_worker
                if config is not None and config.seed is not None
                else None
            ),  # Necessary for reproducibility
            generator=generator,  # Necessary for reproducibility
        )

        if state.rank == 0:
            logger.debug(f"{key} dataloader size: {len(dataloaders[key])} mini-batches")

    return dataloaders
