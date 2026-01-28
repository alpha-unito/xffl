"""Configuration file for the xFFL-LLM example"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms

from xffl.custom.config import DatasetInfo, ModelInfo, XFFLConfig
from xffl.distributed.distributed_state import DistributedState

# Constants
CURRENT_DIR: str = str(os.getcwd())
DATASET_PATH: Path = Path(CURRENT_DIR + "/CIFAR10")


def _get_cifar10_splits() -> Mapping[str, Dataset]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    return {
        "train": datasets.CIFAR10(
            root=DATASET_PATH,
            train=True,
            download=True,
            transform=transform,
        ),
        "val": datasets.CIFAR10(
            root=DATASET_PATH,
            train=False,
            download=True,
            transform=transform,
        ),
    }


def _single_class(
    dataset: Mapping[str, Dataset], config: XFFLConfig, state: DistributedState
):
    if hasattr(config, "one_class") and config.one_class:  # type: ignore
        for _, split in dataset.items():
            split.data = split.data[split.targets == state.rank % 10]  # type: ignore
            split.targets = split.targets[split.targets == state.rank % 10]  # type: ignore


# Model information
@dataclass
class CNN(ModelInfo):
    name: str = "CNN"
    model: Callable = models.resnet18


# Dataset information
@dataclass
class Cifar(DatasetInfo):
    name: str = "CIFAR10"
    splits: Callable = _get_cifar10_splits
    batch_sizes: Mapping[str, int] = field(
        default_factory=lambda: {"train": 64, "val": 1}
    )
    filters: Callable = _single_class
    path: Path = DATASET_PATH


# XFFL configuration
@dataclass
class xffl_config(XFFLConfig):

    # Default
    model_info: ModelInfo = field(default_factory=CNN)
    dataset_info: DatasetInfo = field(default_factory=Cifar)

    # General
    loglevel: int = logging.DEBUG
    seed: int = 42

    # Learning
    learning_rate: float = 1e-2
    epochs: int = 10
    criterion: Callable = field(default_factory=nn.CrossEntropyLoss)

    # WandB
    wandb_entity: str = "alpha-unito"
    wandb_project: str = "xFFL playground"
    wandb_group: str = "02_CNN"
    wandb_name: str = "Example"
    wandb_notes: str = "Example run of xFFL with a CNN"
    wandb_tags: Sequence[str] = field(
        default_factory=lambda: ["xFFL", "example", "MLP"]
    )
    wandb_mode: str = "online"

    # Custom
    one_class: bool = False
    momentum: float = 0.9
