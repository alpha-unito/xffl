"""Configuration file for the xFFL-LLM example"""

import logging
from dataclasses import dataclass

from torch import nn
import torch.nn.functional as F  

from xffl.custom.config_info import DatasetInfo, ModelInfo


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
    path = "/beegfs/home/gmittone/Santimaria/xffl/examples/simulation/01_simple-MLP/data"


@dataclass
class xffl_config:
    model: ModelInfo = Mlp
    dataset: DatasetInfo = Mnist
    loglevel: int = logging.DEBUG
    seed: int = 42
    hsdp: int = 1
    train_batch_size: int = 1024
    val_batch_size: int = 1
    workers: int = 0
    learning_rate: float = 1e-2
    epochs: int = 10
    one_class: bool = False
    # subsampling: int = (5000, 800)
