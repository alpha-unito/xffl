"""Configuration file for the xFFL-LLM example"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms

from xffl.custom.config import DatasetInfo, ModelInfo, OptimizerInfo, XFFLConfig
from xffl.distributed.distributed_state import DistributedState

# Constants
CURRENT_DIR: str = str(os.getcwd()) + "/xffl/examples/intra-silo/02_CNN"
DATASET_PATH: Path = Path(CURRENT_DIR + "/CIFAR10")


# Model information
@dataclass
class CNN(ModelInfo):

    @staticmethod
    def _get_resnet18(config: XFFLConfig, state: DistributedState) -> nn.Module:
        return models.resnet18()

    name: str = "CNN"
    model: Callable = _get_resnet18


# Dataset information
@dataclass
class Cifar(DatasetInfo):

    @staticmethod
    def _get_cifar10_splits(
        config: XFFLConfig, state: DistributedState
    ) -> Mapping[str, Dataset]:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
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

    @staticmethod
    def _single_class(
        dataset: Mapping[str, Dataset], config: XFFLConfig, state: DistributedState
    ) -> Dataset:
        if config.one_class:  # type: ignore
            dataset.data = dataset.data[dataset.targets == state.rank % 10]  # type: ignore
            dataset.targets = dataset.targets[dataset.targets == state.rank % 10]  # type: ignore
        return dataset  # type: ignore

    name: str = "CIFAR10"
    splits: Callable = _get_cifar10_splits
    batch_sizes: Mapping[str, int] = field(
        default_factory=lambda: {"train": 64, "val": 1}
    )
    filters: Callable = _single_class
    path: Path = DATASET_PATH


# Optimizer information
@dataclass
class SGD(OptimizerInfo):
    """Optimizer configuration for BabyLM pretraining."""

    optimizer: Callable = SGD

    optimizer_params: Mapping[str, Any] = field(
        default_factory=lambda: {"lr": 1e-2, "momentum": 0.9}
    )


# XFFL configuration
@dataclass
class xffl_config(XFFLConfig):

    # Default
    model_info: ModelInfo = field(default_factory=CNN)
    dataset_info: DatasetInfo = field(default_factory=Cifar)
    optimizer_info: OptimizerInfo = field(default_factory=SGD)

    # General
    loglevel: int = logging.INFO
    seed: int = 42

    # Learning
    epochs: int = 10
    criterion: Callable = field(default_factory=nn.CrossEntropyLoss)

    # WandB
    wandb_params: Mapping[str, Any] = field(
        default_factory=lambda: {
            "entity": "alpha-unito",
            "project": "xFFL playground",
            "group": "02_CNN",
            "name": "Example",
            "notes": "Example run of xFFL with a CNN",
            "tags": ["xFFL", "example", "MLP"],
            "mode": "online",
        }
    )

    # Custom
    one_class: bool = False
