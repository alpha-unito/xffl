"""Configuration file for the xFFL-LLM example"""

import logging
from dataclasses import dataclass

from torch import nn
import torch.nn.functional as F
from torchvision import models

from xffl.custom.config_info import DatasetInfo, ModelInfo


@dataclass
class Cnn(ModelInfo):
    name = "CNN"
    model_type = nn.Module
    class_ = models.resnet18


@dataclass
class Cifar(DatasetInfo):
    name = "CIFAR10"
    path = "data"


@dataclass
class xffl_config:
    model: ModelInfo = Cnn
    dataset: DatasetInfo = Cifar
    loglevel: int = logging.DEBUG
    seed: int = 42
    wandb_entity: str = "alpha-unito"
    wandb_project: str = "xFFL - convergence"
    wandb_group: str = "test"
    wandb_mode: str = "disabled"
    hsdp: int = 1
    train_batch_size: int = 1024
    val_batch_size: int = 1
    workers: int = 0
    learning_rate: float = 1e-2
    momentum: float = 0.9
    epochs: int = 10
    one_class: bool = False
    # subsampling: int = (5000, 800)
