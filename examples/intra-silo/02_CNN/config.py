"""Configuration file for the xFFL-LLM example"""

import logging
from dataclasses import dataclass, field

from torch import nn
from torchvision import models

from xffl.custom.config import DatasetInfo, ModelInfo, XFFLConfig


# Model information
@dataclass
class CNN(ModelInfo):
    name = "CNN"
    model_type = nn.Module
    class_ = models.resnet18


# Dataset information
@dataclass
class Cifar(DatasetInfo):
    name = "CIFAR10"
    path = "data"


# XFFL configuration
@dataclass
class xffl_config(XFFLConfig):

    # Default
    model_info: ModelInfo = field(default_factory=CNN)
    dataset_info: DatasetInfo = field(default_factory=Cifar)
    loglevel: int = logging.DEBUG
    seed: int = 42

    hsdp: int = 1
    train_batch_size: int = 1024
    val_batch_size: int = 1
    learning_rate: float = 1e-2
    momentum: float = 0.9
    epochs: int = 10
    # subsampling: int = (5000, 800)

    # WandB
    wandb_entity: str = "alpha-unito"
    wandb_project: str = "xFFL - convergence"
    wandb_group: str = "test"
    wandb_mode: str = "disabled"

    # Custom
    one_class: bool = False
