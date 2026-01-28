"""Configuration file for the xFFL simple MLP example"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from xffl.custom.config import DatasetInfo, ModelInfo, XFFLConfig
from xffl.distributed.distributed_state import DistributedState

# Constants
CURRENT_DIR: str = str(os.getcwd())
DATASET_PATH: Path = Path(CURRENT_DIR + "/MNIST")


# Helper methods and classes
class _Model(nn.Module):
    def __init__(self):
        super(_Model, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.log_softmax(self.fc3(x), dim=1)
        return output


def _get_mnist_splits() -> Mapping[str, Dataset]:
    transform: transforms.Compose = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    return {
        "train": datasets.MNIST(
            root=DATASET_PATH,
            train=True,
            download=True,
            transform=transform,
        ),
        "val": datasets.MNIST(
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
class SimpleMLP(ModelInfo):
    name: str = "simple_MLP"
    model: Callable = _Model


# Dataset information
@dataclass
class Mnist(DatasetInfo):
    name: str = "MNIST"
    splits: Callable = _get_mnist_splits
    batch_sizes: Mapping[str, int] = field(
        default_factory=lambda: {"train": 1024, "val": 1}
    )
    filters: Callable = _single_class
    path: Path = DATASET_PATH


# XFFL configuration
@dataclass
class xffl_config(XFFLConfig):

    # Model and dataset
    model_info: ModelInfo = field(default_factory=SimpleMLP)
    dataset_info: DatasetInfo = field(default_factory=Mnist)

    # General
    loglevel: int = logging.DEBUG
    seed: int = 42

    # Learning
    learning_rate: float = 1e-2
    epochs: int = 3
    criterion: nn.Module = field(default_factory=nn.NLLLoss)

    # Custom
    one_class: bool = False

    # WandB
    wandb_entity: str = "alpha-unito"
    wandb_project: str = "xFFL playground"
    wandb_group: str = "01_simple_MLP"
    wandb_name: str = "Example"
    wandb_notes: str = "Example run of xFFL with a simple MLP"
    wandb_tags: Sequence[str] = field(
        default_factory=lambda: ["xFFL", "example", "MLP"]
    )
    wandb_mode: str = "online"
