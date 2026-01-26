"""Configuration file for the xFFL simple MLP example"""

import logging
from dataclasses import dataclass
from typing import Tuple

import torch.nn.functional as F
from torch import nn

from xffl.custom.config_info import DatasetInfo, ModelInfo, XFFLConfig


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.log_softmax(self.fc3(x), dim=1)
        return output


@dataclass
class Mlp(ModelInfo):
    name = "simple_MLP"
    model_type = nn.Module
    class_ = Net


@dataclass
class Mnist(DatasetInfo):
    name = "MNIST"
    path = "data"


@dataclass
class xffl_config(XFFLConfig):

    # Default
    model: ModelInfo = Mlp
    dataset: DatasetInfo = Mnist
    loglevel: int = logging.DEBUG
    seed: int = 42
    train_batch_size: int = 1024
    learning_rate: float = 1e-2
    epochs: int = 3
    hsdp: int = 1
    federated: Tuple[int, ...] = (
        1,
        3,
    )

    # Custom
    one_class: bool = False
