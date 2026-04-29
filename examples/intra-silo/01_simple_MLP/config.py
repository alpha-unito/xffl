"""Configuration file for the xFFL simple MLP example"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import torch.nn.functional as F
from torch import nn
from torch.optim import Adadelta
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from xffl.custom.config import DatasetInfo, ModelInfo, OptimizerInfo, XFFLConfig
from xffl.distributed.distributed_state import DistributedState

# Constants
CURRENT_DIR: str = str(os.getcwd()) + "/xffl/examples/intra-silo/01_simple_MLP"
DATASET_PATH: Path = Path(CURRENT_DIR + "/MNIST")


# Simple MLP
class _Model(nn.Module):
    def __init__(self, config: XFFLConfig, state: DistributedState):
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


# Model information
@dataclass
class SimpleMLP(ModelInfo):
    name: str = "simple_MLP"
    model: Callable = _Model


# Dataset information
@dataclass
class Mnist(DatasetInfo):

    @staticmethod
    def _get_mnist_splits(
        config: XFFLConfig, state: DistributedState
    ) -> Mapping[str, Dataset]:
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

    @staticmethod
    def _single_class(
        dataset: Mapping[str, Dataset], config: XFFLConfig, state: DistributedState
    ) -> Dataset:
        if config.one_class:  # type: ignore
            dataset.data = dataset.data[dataset.targets == state.rank % 10]  # type: ignore
            dataset.targets = dataset.targets[dataset.targets == state.rank % 10]  # type: ignore
        return dataset  # type: ignore

    name: str = "MNIST"
    splits: Callable = _get_mnist_splits
    batch_sizes: Mapping[str, int] = field(
        default_factory=lambda: {"train": 1024, "val": 1}
    )
    filters: Callable = _single_class
    path: Path = DATASET_PATH


# Optimizer information
@dataclass
class AdaDelta(OptimizerInfo):
    """Optimizer configuration for BabyLM pretraining."""

    optimizer: Callable = Adadelta

    optimizer_params: Mapping[str, Any] = field(
        default_factory=lambda: {
            "lr": 1e-2,
        }
    )


# XFFL configuration
@dataclass
class xffl_config(XFFLConfig):

    # Model and dataset
    model_info: ModelInfo = field(default_factory=SimpleMLP)
    dataset_info: DatasetInfo = field(default_factory=Mnist)
    optimizer_info: OptimizerInfo = field(default_factory=AdaDelta)

    # General
    loglevel: int = logging.INFO
    seed: int = 42

    # Learning
    epochs: int = 3
    criterion: nn.Module = field(default_factory=nn.NLLLoss)

    # Custom
    one_class: bool = False

    # WandB
    wandb_params: Mapping[str, Any] = field(
        default_factory=lambda: {
            "entity": "alpha-unito",
            "project": "xFFL playground",
            "group": "01_simple_MLP",
            "name": "Example",
            "notes": "Example run of xFFL with a simple MLP",
            "tags": ["xFFL", "example", "MLP"],
            "mode": "online",
        }
    )
