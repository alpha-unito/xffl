"""xFFL base configuration"""

import functools
import logging
import os
from abc import ABC
from dataclasses import dataclass
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Type

from torch import nn
from torch.distributed.fsdp import wrap

from xffl.distributed.distributed_state import DistributedState

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


@dataclass
class XFFLConfig(ABC):  # type: ignore
    ...


@dataclass
class ModelInfo(ABC):
    """Abstract model information

    Only the model name and instantiation function are mandatory; model name cannot be empty.

    The model path can be overridden at configuration instantiation time if execution is happening inside a container.
    The accepted attetion implementation are: paged|eager, paged|sdpa, paged|flash_attention_2, paged|flash_attention_3, sdpa, flex_attention, flash_attention_2, flash_attention_3.
    If a decoder layer is specified and the wrapping policy is not, a standard wrapping policy will automatically be instantiated based on the provided decoder layer.

    The model configuration is double-chekced before execution.


    :param name: Model's name
    :type name: str
    :param model: Model's class or creation function
    :type model: Callable[[XFFLConfig, DistributedState], nn.Module]
    :param path: Path to the model file or folder, defaults to None
    :type path: Optional[Path | str], optional
    :param attention: Attention layer implementation, defaults to None
    :type attention: Optional[str], optional
    :param decoder_layer: Model's decoder layer class, defatuls to None
    :type decoder_layer: Optional[Type], optional
    :param wrapping_policy: Wrapping policy for FSDP/HSDP, defaults to None
    :type wrapping_policy: Optional[Callable], optional
    :param activation_checkpointing: Activate activation checkpointing, defaults to None
    :type activation_checkpointing: Optional[bool], optional
    :param tokenizer: Tokenizer's class or creation function, defaults to None
    :type tokenizer: Optional[Callable[[XFFLConfig, DistributedState], nn.Module]], optional
    :raises ValueError: If some configuration values are incompatible with their expected characteristics
    """

    # Mandatory
    name: str
    model: Callable[[XFFLConfig, DistributedState], nn.Module]  # type: ignore

    # Optional
    path: Optional[Path | str] = None
    attention: Optional[str] = None
    decoder_layer: Optional[Type] = None
    wrapping_policy: Optional[Callable] = None
    activation_checkpointing: Optional[bool] = None
    tokenizer: Optional[Callable[[XFFLConfig, DistributedState], nn.Module]] = None  # type: ignore

    def __post_init__(self) -> None | ValueError:
        """
        Model configuration validation.

        :raises ValueError: If some configuration values are incompatible with their expected characteristics
        """

        err_msg: str = ""

        # Name
        if not isinstance(self.name, str) or self.name == "":
            err_msg += "Model configuration error: the specified model name is invalid ({self.name}).\n"

        # Model
        if not isinstance(self.model, Callable):
            err_msg += "Model configuration error: the specified model class or instantiation function is not callable ({self.model}).\n"

        # Path
        if self.path is not None:
            if "XFFL_IMAGE" in os.environ:
                logger.debug(
                    'Automatically setting model path to the default one for container execution ("/model/").'
                )
                self.path = Path("/model/")
            else:
                self.path = Path(self.path)

            if not self.path.exists():
                err_msg += "Model configuration error: model path does not exists ({self.path}).\n"

        # Attention
        if self.attention is not None and self.attention not in [
            "paged|eager",
            "paged|sdpa",
            "paged|flash_attention_2",
            "paged|flash_attention_3",
            "sdpa",
            "flex_attention",
            "flash_attention_2",
            "flash_attention_3",
        ]:
            err_msg += "Model configuration error: the specified attention implementation is not supported ({self.attention}).\n"

        # Decoder layer
        if self.decoder_layer is not None and not isinstance(self.decoder_layer, Type):
            err_msg += "Model configuration error: the specified decoder layer is not a type ({self.decoder_layer}).\n"

        # Wrapping policy
        if self.wrapping_policy is not None and not isinstance(
            self.wrapping_policy, Callable
        ):
            err_msg += "Model configuration error: the specified wrapping policy is not callable ({self.tokenizer}).\n"
        elif self.wrapping_policy is None and self.decoder_layer is not None:
            logger.debug(
                f"Automatically setting the wrap policy to the default one based on {self.decoder_layer}."
            )
            self.wrapping_policy = functools.partial(
                wrap.transformer_auto_wrap_policy,
                transformer_layer_cls={self.decoder_layer},
            )

        # Activation Checkpointing
        if self.activation_checkpointing is not None and not isinstance(
            self.activation_checkpointing, bool
        ):
            err_msg += "Model configuration error: the specified activation checkpointing value is not acceptable ({self.tokenizer}).\n"
        elif self.activation_checkpointing and self.decoder_layer is None:
            logger.warning(
                "Model configuration error: activation checkpointing cannot be activated without specifying a decoder layer; setting activation checkpointing to False.\n"
            )
            self.activation_checkpointing = False

        # Tokenizer
        if self.tokenizer is not None and not isinstance(self.tokenizer, Callable):
            err_msg += "Model configuration error: the specified tokenizer is not callable ({self.tokenizer}).\n"

        # Log errors, if any, and raise a ValueError exception
        if not err_msg == "":
            logger.critical(err_msg)
            raise ValueError(err_msg)


@dataclass
class DatasetInfo(ABC):
    """Abstract dataset essential information

    :param splits: Mapping between the split's name and its location inside the dataset's folder
    :type splits: Mapping[str, str]
    :param path: Path to the dataset's folder
    :type path: str
    :param collate_fn: Collate function to apply to the dataloaders
    :type collate_fn: str
    """

    name: str
    splits: Mapping[str, Any]
    batch_sizes: Mapping[str, int]
    filters: Optional[Callable | Sequence[Callable]] = None
    collate_fn: Optional[Callable] = None
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
    gradient_clipping: Optional[float] = None
    gradient_accumulation: Optional[int] = None

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
