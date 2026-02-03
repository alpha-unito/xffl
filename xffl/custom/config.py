"""xFFL base configuration"""

import functools
import logging
import os
from abc import ABC
from dataclasses import dataclass
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Type

import torch
from torch import nn
from torch.distributed.distributed_c10d import Backend
from torch.distributed.fsdp import MixedPrecision, wrap
from torch.optim import Optimizer
from transformers import AutoTokenizer

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
    :param activation_checkpointing: Activate activation checkpointing, defaults to False
    :type activation_checkpointing: Optional[bool], optional
    :param tokenizer: Tokenizer's class or creation function, defaults to None
    :type tokenizer: Optional[Callable[[XFFLConfig, DistributedState], nn.Module]], optional
    :param mixed_precision: Mixed precision configuration, defaults to None
    :type mixed_precision: Optional[MixedPrecision], optional
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
    tokenizer: Optional[Callable[[XFFLConfig, DistributedState], AutoTokenizer]] = None  # type: ignore
    mixed_precision: Optional[MixedPrecision] = None

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
        if self.activation_checkpointing is not None:
            if not isinstance(self.activation_checkpointing, bool):
                err_msg += f"Model configuration error: the specified activation checkpointing value is not acceptable ({self.tokenizer}).\n"
            elif self.activation_checkpointing and self.decoder_layer is None:
                logger.warning(
                    "Model configuration error: activation checkpointing cannot be activated without specifying a decoder layer.\n"
                )

        # Tokenizer
        if self.tokenizer is not None and not isinstance(self.tokenizer, Callable):
            err_msg += f"Model configuration error: the specified tokenizer is not callable ({self.tokenizer}).\n"

        # Mixed precision
        if self.mixed_precision is not None and not isinstance(
            self.mixed_precision, MixedPrecision
        ):
            err_msg += f"Model configuration error: the provided mixed precision configuration is not valid ({self.mixed_precision}).\n"

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
    collate_fn: Optional[Callable | Mapping[str, Callable]] = None
    filters: Optional[
        Callable
        | Sequence[Callable]
        | Mapping[str, Callable]
        | Mapping[str, Sequence[Callable]]
    ] = None

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

            if not self.path.parent.exists():
                err_msg += f"Dataset configuration error: dataset path does not exists ({self.path}).\n"

        # Workers
        if self.workers is not None and (
            not isinstance(self.workers, int) or self.workers < 0
        ):
            err_msg += f"Dataset configuration error: the specified workers number is invalid ({self.workers}).\n"

        # Subsampling
        if self.subsampling is not None:
            if isinstance(self.subsampling, int) and self.subsampling < 0:
                err_msg += f"Dataset configuration error: the specified subsampling configuration is a negative integer ({self.subsampling}).\n"
            elif isinstance(self.subsampling, Mapping) and not all(
                subsample > 0 for subsample in self.subsampling.values()
            ):
                err_msg += f"Dataset configuration error: the specified subsampling configuration is not a mapping of split names and positive integers ({self.subsampling}).\n"
            elif not (
                isinstance(self.subsampling, int)
                or isinstance(self.subsampling, Mapping)
            ):
                err_msg += f"Dataset configuration error: the specified subsampling configuration is not an integer or a mapping of split names and integers ({self.subsampling}).\n"

        # Batch size
        if self.batch_sizes is not None:
            if not (
                isinstance(self.batch_sizes, int)
                or isinstance(self.batch_sizes, Mapping)
            ):
                err_msg += f"Dataset configuration error: the specified batch size configuration is not an integer or a mapping of split names and integers ({self.batch_sizes}).\n"
            elif isinstance(self.batch_sizes, int) and self.batch_sizes < 0:
                err_msg += f"Dataset configuration error: the specified batch size configuration is a negative integer ({self.batch_sizes}).\n"
            elif isinstance(self.batch_sizes, Mapping) and not all(
                subsample > 0 for subsample in self.batch_sizes.values()
            ):
                err_msg += f"Dataset configuration error: the specified batch size configuration is not a mapping of split names and positive integers ({self.batch_sizes}).\n"

        # Collate function
        if self.collate_fn is not None:
            if not (
                isinstance(self.collate_fn, Callable)
                or isinstance(self.collate_fn, Mapping)
            ):
                err_msg += f"Dataset configuration error: the specified collate function is not callable or a mapping of split names and callables ({self.collate_fn}).\n"
            elif isinstance(self.collate_fn, Mapping) and not all(
                isinstance(fn, Callable) for fn in self.collate_fn.values()
            ):
                err_msg += f"Dataset configuration error: the specified collate functions are not a mapping of split names and callables ({self.collate_fn}).\n"

        # Filters
        if self.filters is not None:
            if not (
                isinstance(self.filters, Callable)
                or isinstance(self.filters, Sequence)
                or isinstance(self.filters, Mapping)
            ):
                err_msg += f"Dataset configuration error: the specified filter functions are not callable, a sequence of callables or mapping of split names and callables or sequence of callables ({self.filters}).\n"
            elif isinstance(self.filters, Sequence) and not all(
                isinstance(fil, Callable) for fil in self.filters
            ):
                err_msg += f"Dataset configuration error: the specified filters are not all callable ({self.filters}).\n"
            elif isinstance(
                self.filters, Mapping
            ):  # TODO: Check callable in case of Mapping[Sequence[Callable]]
                if not (
                    (isinstance(fil, Callable) for fil in self.filters.values())
                    or all(isinstance(fil, Sequence) for fil in self.filters.values())
                ):
                    err_msg += f"Dataset configuration error: the specified filters map is not a mapping of split names and callables or sequence of callables ({self.filters}).\n"

        # Log errors, if any, and raise a ValueError exception
        if not err_msg == "":
            logger.critical(err_msg)
            raise ValueError(err_msg)


@dataclass
class XFFLConfig(ABC):
    """Base xFFL configuration.

    Only the model_info, dataset_info, and optimizer fields are mandatory.

    If multiple process constitute the xFFL world but no parallelization strategy is specified, FSDP will be instantiated over all the available processes.
    If federated is specified, equal groups of federated processes will be instantiated; if federated is a sequence, the processes will be divided into federated groups according to such values.
    In case of equal (symmtrical) federated groups, both FSDP and HSDP can be instantiated, while with unequal (asymmetrical) federated groups only HSDP is supported (given that the HSDP replica group size divides the federated group sizes, in both scenarios).
    The federated_batches and cuda_streams parameters are effective only if a valid federated value is provided, thus setting up Federated Scaling.

    To save the trained model, both output_folder and output_model should be specified. The output model name cannot be empty.
    The output path can be overridden at configuration instantiation time if execution is happening inside a container.

    The xFFL configuration is double-checked before execution.

    :param model_info: Model configuration
    :type model_info: ModelInfo
    :param dataset_info: Dataset configuration
    :type dataset_info: DatasetInfo
    :param optimizer: Optimizer creation function
    :type optimizer: Callable[[nn.Module, XFFLConfig], Optimizer]
    :param loglevel: Logging level, expressed as an integer following the python logging package, defaults to INFO (20)
    :type loglevel: int
    :param seed: Random number generator seed, essential for reproducible execution, defaults to None
    :type seed: Optional[int], optional
    :param hsdp: Activate HSDP with the specified replica group size, defaults to None
    :type hsdp: Optional[int], optional
    :param federated: Activate FederatedScaling with the specified federated group sizes, defaults to None
    :type federated: Optional[int | Sequence[int]], optional
    :param federated_batches: Specified after how many local batches the aggregation between the federated groups should be run, defaults to None
    :type federated_batches: Optional[int], optional
    :param cuda_streams: Number of CUDA streams to instantiate in case FederatedScaling is setup, defaults to None
    :type cuda_streams: Optional[int], optional
    :param epochs: Number of epochs, defaults to None
    :type epochs: Optional[int], optional
    :param learning_rate: Learning rate, defaults to None
    :type learning_rate: Optional[float], optional
    :param lr_scheduler: Learning rate scheduler, defaults to None
    :type lr_scheduler: Optional[Callable], optional
    :param scale_learning_rate: If to scale the learning rate in the number of available processes, defaults to None
    :type scale_learning_rate: Optional[bool], optional
    :param criterion: Loss function, defaults to None
    :type criterion: Optional[Callable], optional
    :param gradient_clipping: Value to clip the gradient to, defaults to None
    :type gradient_clipping: Optional[float], optional
    :param gradient_accumulation: Number of steps of gradient accumulation before running an optimization step, defaults to None
    :type gradient_accumulation: Optional[int], optional
    :param output_folder: Output folder path where to save the trained model, defaults to None
    :type output_folder: Optional[Path], optional
    :param output_model: Saving name for the trained model, defaults to None
    :type output_model: Optional[str], optional
    :param wandb_entity: WandB entity, defaults to None
    :type wandb_entity: Optional[str], optional
    :param wandb_project: WandB project, defaults to None
    :type wandb_project: Optional[str], optional
    :param wandb_group: WandB run group, defaults to None
    :type wandb_group: Optional[str], optional
    :param wandb_name: WandB run name, defaults to None
    :type wandb_name: Optional[str], optional
    :param wandb_notes: WandB run notes, defaults to None
    :type wandb_notes: Optional[str], optional
    :param wandb_tags: WandB run tags, defaults to None
    :type wandb_tags: Optional[Sequence[str]], optional
    :param wandb_mode: WandB execution mode, defaults to None
    :type wandb_mode: Optional[Literal["online", "offline", "disabled", "shared"]], optional
    :raises ValueError: If some configuration values are incompatible with their expected characteristics
    """

    # Mandatory - Model and dataset
    model_info: ModelInfo
    dataset_info: DatasetInfo
    optimizer: Callable[[nn.Module, XFFLConfig], Optimizer]

    # General
    loglevel: int = logging.INFO
    seed: Optional[int] = None

    # Distributed training
    hsdp: Optional[int] = None
    federated: Optional[int | Sequence[int]] = None
    federated_batches: Optional[int] = None
    cuda_streams: Optional[int] = None

    # Learning
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    lr_scheduler: Optional[Callable] = None
    scale_learning_rate: Optional[bool] = None
    criterion: Optional[nn.Module] = None
    gradient_clipping: Optional[float] = None
    gradient_accumulation: Optional[int] = None

    # Output
    output_folder: Optional[Path] = None
    output_model: Optional[str] = None

    # WandB
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_notes: Optional[str] = None
    wandb_tags: Optional[Sequence[str]] = None
    wandb_mode: Optional[Literal["online", "offline", "disabled", "shared"]] = None

    # Manual distributed configuration
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

    def __post_init__(self) -> None | ValueError:
        """
        xFFL configuration validation.

        :raises ValueError: If some configuration values are incompatible with their expected characteristics
        """

        err_msg: str = ""

        # Model info
        if not isinstance(self.model_info, ModelInfo):
            err_msg += f"xFFL configuration error: the provided model configuration is not an instance of ModelInfo ({self.model_info}).\n"

        # Dataset info
        if not isinstance(self.dataset_info, DatasetInfo):
            err_msg += f"xFFL configuration error: the provided dataset configuration is not an instance of ModelInfo ({self.dataset_info}).\n"

        # Optimizer
        if not isinstance(self.optimizer, Callable):
            err_msg += f"xFFL configuration error: the provided optimizer creation function is not callable ({self.optimizer}).\n"

        # Log level
        if not isinstance(self.loglevel, int) or self.loglevel < 0:
            err_msg += f"xFFL configuration error: the provided logging level is not valid ({self.loglevel}).\n"

        # Seed
        if self.seed is not None and not isinstance(self.seed, int):
            err_msg += f"xFFL configuration error: the provided seed is not an integer ({self.seed}).\n"

        # HSDP
        if self.hsdp is not None and (not isinstance(self.hsdp, int) or self.hsdp < 0):
            err_msg += f"xFFL configuration error: the provided hsdp replica group size is not valid ({self.hsdp}).\n"

        # Federated
        if self.federated is not None:
            if not (
                isinstance(self.federated, int) or isinstance(self.federated, Sequence)
            ):
                err_msg += f"xFFL configuration error: the provided federated group size is not valid ({self.federated}).\n"
            elif isinstance(self.federated, int) and self.federated < 0:
                err_msg += f"xFFL configuration error: the provided federated group size is not a positive integer ({self.federated}).\n"
            elif isinstance(self.federated, Sequence) and not all(
                fed > 0 for fed in self.federated
            ):
                err_msg += f"xFFL configuration error: the provided federated group sizes are not all positive integers ({self.federated}).\n"

        # Federated batches
        if self.federated_batches is not None:
            if self.federated is None:
                logger.warning(
                    f"xFFL configuration error: the provided federated batches value will be ignored since FederatedScaling is not setup ({self.federated_batches}).\n"
                )
                self.federated = None
            elif (
                not isinstance(self.federated_batches, int)
                or self.federated_batches < 0
            ):
                err_msg += f"xFFL configuration error: the provided federated batches value is not valid ({self.federated_batches}).\n"

        # CUDA streams
        if self.cuda_streams is not None:
            if self.federated is None:
                logger.warning(
                    f"xFFL configuration error: the provided CUDA streams number will be ignored since FederatedScaling is not setup ({self.federated_batches}).\n"
                )
                self.cuda_streams = None
            elif not isinstance(self.cuda_streams, int) or self.cuda_streams < 0:
                err_msg += f"xFFL configuration error: the provided CUDA streams number is not valid ({self.cuda_streams}).\n"

        # Epochs
        if self.epochs is not None and (
            not isinstance(self.epochs, int) or self.epochs < 0
        ):
            err_msg += f"xFFL configuration error: the provided epochs value is not valid ({self.epochs}).\n"

        # Learning rate
        if self.learning_rate is not None and not isinstance(self.learning_rate, float):
            err_msg += f"xFFL configuration error: the provided learning rate is not valid ({self.learning_rate}).\n"

        # Learning rate scheduler
        if self.lr_scheduler is not None and not isinstance(
            self.lr_scheduler, Callable
        ):
            err_msg += f"Model configuration error: the specified learning rate scheduler is not callable ({self.lr_scheduler}).\n"

        # Scale learning rate
        if self.scale_learning_rate is not None and not isinstance(
            self.scale_learning_rate, bool
        ):
            err_msg += f"Model configuration error: the specified value of scale_learning_rate is not acceptable ({self.scale_learning_rate}).\n"

        # Criterion
        if self.criterion is not None and not isinstance(self.criterion, Callable):
            err_msg += f"Model configuration error: the specified criterion is not callable ({self.criterion}).\n"

        # Gradient clipping
        if self.gradient_clipping is not None and not isinstance(
            self.gradient_clipping, float
        ):
            err_msg += f"xFFL configuration error: the provided gradient clipping value is not valid ({self.gradient_clipping}).\n"

        # Gradient accumulation
        if self.gradient_accumulation is not None and (
            not isinstance(self.gradient_accumulation, int)
            or self.gradient_accumulation < 0
        ):
            err_msg += f"xFFL configuration error: the provided gradient accumulation value is not valid ({self.gradient_accumulation}).\n"

        # Output folder
        if self.output_folder is not None:
            if "XFFL_IMAGE" in os.environ:
                logger.debug(
                    'Automatically setting output folder path to the default one for container execution ("/output/").'
                )
                self.output_folder = Path("/output/")
            else:
                self.output_folder = Path(self.output_folder)

            if not self.output_folder.exists():
                err_msg += f"xFFL configuration error: the output folder path does not exists ({self.output_folder}).\n"

        # Output model
        if (
            self.output_model is not None
            and not isinstance(self.output_model, str)
            or self.output_model == ""
        ):
            err_msg += f"xFFL configuration error: the specified output model name is invalid ({self.output_model}).\n"

        if (self.output_folder is None and self.output_model is not None) or (
            self.output_folder is not None and self.output_model is None
        ):
            logger.warning(
                f"xFFL configuration error: the output model folder ({self.output_folder}) and name ({self.output_model}) are not both correctly specified; the model will not be saved."
            )
            self.output_folder, self.output_model = None, None

        # Log errors, if any, and raise a ValueError exception
        if not err_msg == "":
            logger.critical(err_msg)
            raise ValueError(err_msg)


# TODO: tokenizer + collate_fn
