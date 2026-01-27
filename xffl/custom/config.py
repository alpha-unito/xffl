"""Abstract dataset class"""

import logging
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Tuple, Type


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

    name: str
    model: Callable
    decoder_layers: Optional[Type] = None
    wrapping_policy: Optional[Callable] = None
    path: Optional[Path] = None


@dataclass
class DatasetInfo(ABC):
    """Abstract dataset essential information

    :param splits: Mapping between the split's name and its location inside the dataset's folder
    :type splits: Mapping[str, str]
    :param path: Path to the dataset's folder
    :type path: str
    """

    name: str
    splits: Mapping[str, Any]
    batch_sizes: Mapping[str, int]
    filters: Optional[Callable | Tuple[Callable, ...]] = None
    subsampling: Optional[int | Tuple[int, ...]] = None
    workers: Optional[int] = None
    path: Optional[Path | str] = None


@dataclass
class XFFLConfig(ABC):
    """Abstract base XFFL configuration

    :param model: Model configuration
    :type model: ModelInfo
    :param dataset: Dataset configuration
    :type dataset: DatasetInfo
    """

    # Mandatory - Model and dataset
    model_info: ModelInfo
    dataset_info: DatasetInfo

    # General
    loglevel: int = logging.INFO
    seed: Optional[int] = None

    # Distributed training
    hsdp: Optional[int] = None
    federated: Optional[int | Tuple[int, ...]] = None
    federated_batches: Optional[int] = None
    cuda_streams: Optional[int] = None

    # WandB
    wandb_name: Optional[str] = None
    wandb_mode: Optional[str] = "disabled"

    # Learning
    learning_rate: float = 1e-3
    scale_learning_rate: bool = False
    epochs: int = 1
    attention: str = "sdpa"
    criterion: Optional[Callable] = None

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
    backend: Optional[str] = None
    master_addr: Optional[str] = None
    master_port: Optional[int] = None
    device: Optional[str] = None
