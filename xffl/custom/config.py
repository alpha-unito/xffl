"""Abstract dataset class"""

import logging
import os
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, Sequence


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
    tokenizer: Optional[Callable] = None
    collate_fn: Optional[Callable] = None
    decoder_layers: Optional[type] = None
    wrapping_policy: Optional[Callable | Sequence[Callable]] = None
    attention: Optional[str] = None
    path: Optional[Path | str] = None

    def __post_init__(self):
        if "XFFL_IMAGE" in os.environ:
            self.path = Path("/model/")


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
    filters: Optional[Callable | Sequence[Callable]] = None
    subsampling: Optional[int | Sequence[int]] = None
    workers: Optional[int] = None
    path: Optional[Path | str] = None

    def __post_init__(self):
        if "XFFL_IMAGE" in os.environ:
            self.path = Path("/dataset/")


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
    federated: Optional[int | Sequence[int]] = None
    federated_batches: Optional[int] = None
    cuda_streams: Optional[int] = None

    # WandB
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_notes: Optional[str] = None
    wandb_tags: Optional[Sequence[str]] = None
    wandb_mode: Literal["online", "offline", "disabled", "shared"] = "disabled"

    # Learning
    learning_rate: float = 1e-3
    scale_learning_rate: bool = False
    epochs: int = 1
    criterion: Optional[Callable] = None

    # Output
    output_folder: Optional[Path] = None
    output_model: Optional[str] = None

    # Advanced configuration
    rank: Optional[int] = None
    world_size: Optional[int] = None
    group_local_rank: Optional[int] = None
    group_local_size: Optional[int] = None
    group_rank: Optional[int] = None
    group_world_size: Optional[int] = None
    backend: Optional[str] = None  # torch.distributed.distributed_c10d.Backend
    master_addr: Optional[str] = None
    master_port: Optional[int] = None
    device: Optional[str] = None  # torch.device
    mixed_precision: Optional[Any] = None  # torch.distributed.fsdp.MixedPrecision
    lr_scheduler: Optional[Callable] = None
