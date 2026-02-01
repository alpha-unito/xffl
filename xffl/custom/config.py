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
    """Default model information.

    Only the model name and instantiation function are mandatory; model name cannot be empty.

    The model path can be overridden at configuration instantiation time if execution is happening inside a container.
    The accepted attetion implementation are: paged|eager, paged|sdpa, paged|flash_attention_2, paged|flash_attention_3, sdpa, flex_attention, flash_attention_2, flash_attention_3.
    If a decoder layer is specified and the wrapping policy is not, a standard wrapping policy will automatically be instantiated based on the provided decoder layer.

    The model configuration is double-checked before execution.

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
            err_msg += f"Model configuration error: the specified model name is invalid ({self.name}).\n"

        # Model
        if not isinstance(self.model, Callable):
            err_msg += f"Model configuration error: the specified model class or instantiation function is not callable ({self.model}).\n"

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
                err_msg += f"Model configuration error: model path does not exists ({self.path}).\n"

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
            err_msg += f"Model configuration error: the specified attention implementation is not supported ({self.attention}).\n"

        # Decoder layer
        if self.decoder_layer is not None and not isinstance(self.decoder_layer, Type):
            err_msg += f"Model configuration error: the specified decoder layer is not a type ({self.decoder_layer}).\n"

        # Wrapping policy
        if self.wrapping_policy is not None and not isinstance(
            self.wrapping_policy, Callable
        ):
            err_msg += f"Model configuration error: the specified wrapping policy is not callable ({self.tokenizer}).\n"
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
            err_msg += f"Model configuration error: the specified activation checkpointing value is not acceptable ({self.tokenizer}).\n"
        elif self.activation_checkpointing and self.decoder_layer is None:
            logger.warning(
                "Model configuration error: activation checkpointing cannot be activated without specifying a decoder layer.\n"
            )

        # Tokenizer
        if self.tokenizer is not None and not isinstance(self.tokenizer, Callable):
            err_msg += f"Model configuration error: the specified tokenizer is not callable ({self.tokenizer}).\n"

        # Log errors, if any, and raise a ValueError exception
        if not err_msg == "":
            logger.critical(err_msg)
            raise ValueError(err_msg)


@dataclass
class DatasetInfo(ABC):
    """Default dataset information.

    Only the dataset name and split instantiation function are mandatory; dataset name cannot be empty.

    The model path can be overridden at configuration instantiation time if execution is happening inside a container.
    For subsampling, batch_size, collate_fn, and filater, if a single value is specified, it will be applied to all available dataset splits.
    All mappings are assume a dictionary of the form {"split_name": Any}.

    The model configuration is double-checked before execution.

    :param name: Dataset's name
    :type name: str
    :param splits: Dataset splits' class or creation function
    :type splits: Callable[[XFFLConfig, DistributedState], Mapping[str, Any]]
    :param path: Path to the dataset file or folder, defaults to None
    :type path: Optional[Path | str], optional
    :param workers: Number of worker processes to spawn to load data, defaults to None
    :type workers: Optional[int], optional
    :param subsampling: Number of samples to subsample from each data split, defaults to None
    :type subsampling: Optional[Mapping[str, int]], optional
    :param batch_sizes: Batch size to use for each data split, defaults to None
    :type batch_sizes: Optional[Mapping[str, int]], optional
    :param collate_fn: Collate function to apply to the dataloaders, defaults to None
    :type collate_fn: Optional[Callable | Mapping[str, Callable]], optional
    :param filters: Filter functions to apply before creating the dataloaders, defaults to None
    :type filters: Optional[Callable | Sequence[Callable] | Mapping[str, Callable] | Mapping[str, Sequence[Callable]]], optional
    :raises ValueError: If some configuration values are incompatible with their expected characteristics
    """

    # Mandatory
    name: str
    splits: Callable[[XFFLConfig, DistributedState], Mapping[str, Any]]

    # Optional
    path: Optional[Path | str] = None
    workers: Optional[int] = None
    subsampling: Optional[int | Mapping[str, int]] = None
    batch_sizes: Optional[int | Mapping[str, int]] = None
    collate_fn: Optional[Callable | Mapping[str, Callable]] = (
        None  # TODO: implement this framework-side
    )
    filters: Optional[
        Callable
        | Sequence[Callable]
        | Mapping[str, Callable]
        | Mapping[str, Sequence[Callable]]
    ] = None  # TODO: implement this framework-side

    def __post_init__(self) -> None | ValueError:
        """
        Dataset configuration validation.

        :raises ValueError: If some configuration values are incompatible with their expected characteristics
        """

        err_msg: str = ""

        # Name
        if not isinstance(self.name, str) or self.name == "":
            err_msg += f"Dataset configuration error: the specified dataset name is invalid ({self.name}).\n"

        # Splits
        if not isinstance(self.splits, Callable):
            err_msg += f"Dataset configuration error: the specified splits instantiation function is not callable ({self.splits}).\n"

        # Path
        if self.path is not None:
            if "XFFL_IMAGE" in os.environ:
                logger.debug(
                    'Automatically setting dataset path to the default one for container execution ("/dataset/").'
                )
                self.path = Path("/dataset/")
            else:
                self.path = Path(self.path)

            if not self.path.exists():
                err_msg += f"Dataset configuration error: dataset path does not exists ({self.path}).\n"

        # Workers
        if self.workers is not None and (
            not isinstance(self.workers, int) or self.workers < 0
        ):
            err_msg += f"Dataset configuration error: the specified workers number is invalid ({self.workers}).\n"

        # Subsampling
        if self.subsampling is not None and not (
            isinstance(self.subsampling, int) or isinstance(self.subsampling, Mapping)
        ):
            err_msg += f"Dataset configuration error: the specified subsampling configuration is not an integer or a mapping of split names and integers ({self.subsampling}).\n"

        # Batch size
        if self.batch_sizes is not None and not (
            isinstance(self.batch_sizes, int) or isinstance(self.batch_sizes, Mapping)
        ):
            err_msg += f"Dataset configuration error: the specified batch size configuration is not an integer or a mapping of split names and integers ({self.batch_sizes}).\n"

        # Collate function
        if self.collate_fn is not None and not (
            isinstance(self.collate_fn, Callable)
            or isinstance(self.collate_fn, Mapping)
        ):
            err_msg += f"Dataset configuration error: the specified collate function is not callable ({self.collate_fn}).\n"

        # Filters
        if self.filters is not None and not (
            isinstance(self.filters, Callable)
            or isinstance(self.filters, Sequence)
            or isinstance(self.filters, Mapping)
        ):
            err_msg += f"Dataset configuration error: the specified filters functions are not callable or mapping of split names and callable ({self.filters}).\n"

        # Log errors, if any, and raise a ValueError exception
        if not err_msg == "":
            logger.critical(err_msg)
            raise ValueError(err_msg)


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
