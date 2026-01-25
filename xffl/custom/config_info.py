"""Abstract dataset class"""

import logging
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Tuple, Type

import torch
from torch.distributed.distributed_c10d import Backend


@dataclass
class ModelInfo(ABC):
    """Abstract model essential information

    :param model_type: Class of the selected model
    :type model_type: Type
    :param decoder_layer: Class of the decoder layer of the model
    :type decoder_layer: Type
    :param wrapping_policy: FSDP wrapping function for the decoder layer
    :type wrapping_policy: Callable
    :param path: Path to the model's folder
    :type path: str
    """

    model_type: str
    decoder_layers: Type
    wrapping_policy: Callable
    name: str
    path: str


@dataclass
class DatasetInfo(ABC):
    """Abstract dataset essential information

    :param splits: Mapping between the split's name and its location inside the dataset's folder
    :type splits: Mapping[str, str]
    :param path: Path to the dataset's folder
    :type path: str
    """

    name: str
    splits: Mapping[str, str]
    path: str


@dataclass
class XFFLConfig(ABC):
    """Abstract base XFFL configuration

    :param model: Model configuration
    :type model: ModelInfo
    :param dataset: Dataset configuration
    :type dataset: DatasetInfo
    """

    # Mandatory - Model and dataset
    model: ModelInfo
    dataset: DatasetInfo

    # General
    loglevel: int = logging.INFO
    seed: Optional[int] = None

    # Distributed training
    hsdp: Optional[int] = None
    federated: Optional[int | Tuple[int, ...]] = None
    federated_batches: int = 1
    cuda_streams: Optional[int] = None

    # WandB
    wandb_name: Optional[str] = None
    wandb_mode: Optional[str] = "disabled"

    # Learning
    learning_rate: float = 1e-3
    scale_learning_rate: bool = False
    epochs: int = 1
    train_batch_size: int = 1
    val_batch_size: int = 1
    attention: str = "sdpa"

    # Data
    subsampling: Optional[int] = None
    workers: int = 0

    # Output
    output_folder: Optional[Path] = None
    output_model: Optional[str] = None

    # Misc
    online: bool = False

    # Advanced configuration
    rank: Optional[int] = None
    world_size: Optional[int] = None
    group_local_rank: Optional[int] = None
    group_local_size: Optional[int] = None
    group_rank: Optional[int] = None
    group_world_size: Optional[int] = None
    backend: Optional[Backend] = None
    master_addr: Optional[str] = None
    master_port: Optional[int] = None
    device: Optional[torch.device] = None
