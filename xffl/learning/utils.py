"""Utility methods useful in a number of common DNN training scenarios"""

import functools
import os
import random
import subprocess
import sys
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Type

import numpy
import torch
import torch.nn as nn
import wandb
from torch import Generator
from torch import distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from transformers import PreTrainedModel

from xffl.custom.config import XFFLConfig
from xffl.distributed.distributed_state import DistributedState
from xffl.utils.utils import resolve_param

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def set_deterministic_execution(
    seed: Optional[int] = None, config: Optional[XFFLConfig] = None
) -> Optional[Generator]:  # TODO: Flash_attention is stochastic
    """Set all the necessary RNGs to obtain reproducible executions

    This method sets random, numpy, torch and CUDA RNGs with the same seed.
    It also forces PyTorch's to use deterministic algorithms, reducing performance

    The seed can be provided both directly and through an XFFL configuration.
    In case both are provided, the first takes the precedence.

    :param seed: Random seed
    :type seed: Optional[int], defaults to None
    :param config: XFFL configuration
    :type config: Optional[XFFLConfig], defaults to None
    :return: PyTorch RNG if a seed is provided, else None
    :rtype: Optional[Generator]
    """

    # Parameters resolution
    _seed: Optional[int] = resolve_param(value=seed, config=config, attr="seed")

    generator: Optional[Generator] = None

    if _seed is not None:
        logger.debug(f"Setting RNGs seed to {_seed}")

        random.seed(_seed)
        numpy.random.seed(_seed)
        generator = torch.manual_seed(_seed)
        torch.cuda.manual_seed_all(_seed)

        torch.utils.deterministic.fill_uninitialized_memory = (  # type: ignore
            True  # This should be True by default
        )
        torch.use_deterministic_algorithms(mode=True)

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # TODO: check cuBLAS version
    else:
        logger.warning("No seed provided - deterministic execution will not be set.")

    return generator


def set_nondeterministic_execution() -> None:
    """Deactivate deterministic execution and deterministic memory filling to improve performance"""
    logger.debug("Setting PyTorch deterministic execution")
    torch.utils.deterministic.fill_uninitialized_memory = False  # type: ignore
    torch.use_deterministic_algorithms(mode=False)


def get_model_size(model: nn.Module) -> int:
    """Returns the model's trainable parameters number

    :param model: PyTorch model
    :type model: nn.Module
    :return: Number of trainable parameters
    :rtype: int
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_in_bits(model: nn.Module) -> int:
    """Returns the model's trainable parameters size in bits

    :param model: PyTorch model
    :type model: nn.Module
    :return: Size of trainable parameters in bits
    :rtype: int
    """
    return sum(
        p.numel()
        * (
            torch.finfo(p.data.dtype).bits
            if p.data.is_floating_point()
            else torch.iinfo(p.data.dtype).bits
        )
        for p in model.parameters()
        if p.requires_grad
    )


def seed_dataloader_worker(worker_id: int) -> None:
    """Seeds PyTorch's data loader workers to guarantee reproducibility

    Since each data loader worker is a process, the underlying libraries have to be re-seeded in a reproducible way to guarantee reproducibility of the loaded data and that different workers do not have the same seed, thus loading the same data

    :param worker_id: Worker's id
    :type worker_id: int
    """
    worker_seed = (
        torch.initial_seed() % 2**32
    )  # Get PyTorch generated seed for the worker (base_seed + worker_id) and reduce it into a valid range
    random.seed(worker_seed)
    numpy.random.seed(worker_seed)


def set_activation_checkpointing(
    model: nn.Module | PreTrainedModel, layer: Type
) -> None:
    """Sets up activation (gradient) checkpointing

    This feature reduces maximum memory usage trading off more compute

    :param model: Model on which setting up the checkpointing
    :type model: nn.Module | PreTrainedModel
    :param layer: Layers to wrap, needed only by Torch models, defaults to None
    :type layer: Optional[nn.Module], optional
    """
    if isinstance(model, PreTrainedModel):
        # Specific for HuggingFace models
        # model.enable_input_require_grads()  # TODO: fine-tuning specific?
        try:
            model.gradient_checkpointing_enable()
        except ValueError as e:
            logger.exception(e)
        else:
            logger.debug("Activated reentrant model (gradient) checkpointing")
    else:
        # Generic PyTorch models - non-reentrant checkpointing
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=lambda submodule: isinstance(submodule, layer),
        )
        logger.debug("Activated non-reentrant model (gradient) checkpointing")


def preload(files: Sequence[Optional[Path | str]]) -> None:
    """Pre-loads the given list of files and folders

    Particularly useful on HPC, where data can be moved near the computing nodes ahead of time

    :param files: Paths of the files and folders to be preloaded
    :type files: List[Path|str]
    :raises OSError, ValueError: If the subprocess run fails
    """
    for file in files:
        if file is not None:
            _file: Path = file if isinstance(file, Path) else Path(file)
            logger.debug(f"Preloading: {file}")
            command = " ".join(
                [
                    "find",
                    str(_file),
                    "-type",
                    "f",
                    "-exec",
                    "cat",
                    "{}",
                    "+",
                    ">",
                    "/dev/null",
                    "&",
                ]
            )
            try:
                subprocess.Popen(
                    command,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    shell=True,
                    universal_newlines=True,
                )
            except (OSError, ValueError) as e:
                raise e


def cuda_reset_memory_stats_and_empty_cache() -> None:
    """Reset CUDA peak memory stats and empty CUDA cache.
    This method has no effect if CUDA is not available.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def cuda_sync_and_empty_cache() -> None:
    """Synchronizes CUDA streams with the CPU state and empty CUDA cache.
    This method has no effect if CUDA is not available.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def cuda_sync() -> None:
    """Synchronizes CUDA streams with the CPU state.
    This method has no effect if CUDA is not available.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def barrier(state: DistributedState) -> None:
    """Implements a barrier.
    This method has no effect if the distributed backend is not initialized
    """
    if torch.distributed.is_initialized():
        dist.barrier(device_ids=[state.node_local_rank])


def wandb_setup(
    entity: Optional[str] = None,
    project: Optional[str] = None,
    group: Optional[str] = None,
    name: Optional[str] = None,
    notes: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    mode: Optional[Literal["online", "offline", "disabled", "shared"]] = None,
    config: Optional[XFFLConfig] = None,
) -> Any:
    """Initializes a WandB run.

    :param entity: WandB entity, defaults to None
    :type entity: Optional[str], optional
    :param project: WandB project, defaults to None
    :type project: Optional[str], optional
    :param group: WandB run group, defaults to None
    :type group: Optional[str], optional
    :param name: WandB run name, defaults to None
    :type name: Optional[str], optional
    :param notes: WandB run notes, defaults to None
    :type notes: Optional[str], optional
    :param tags: WandB run tags, defaults to None
    :type tags: Optional[Sequence[str]], optional
    :param mode: WandB execution mode, defaults to None
    :type mode: Optional[Literal["online", "offline", "disabled", "shared"]], optional
    :param config: xFFL configuration, defaults to None
    :type config: Optional[XFFLConfig], optional
    :return: An instantiated WandB run
    :rtype: Any
    """
    # Resolve parameters
    _entity: Optional[str] = resolve_param(
        value=entity, config=config, attr="wandb_entity"
    )
    _project: Optional[str] = resolve_param(
        value=project, config=config, attr="wandb_project"
    )
    _group: Optional[str] = resolve_param(
        value=group, config=config, attr="wandb_group"
    )
    _name: Optional[str] = resolve_param(value=name, config=config, attr="wandb_name")
    _notes: Optional[str] = resolve_param(
        value=notes, config=config, attr="wandb_notes"
    )
    _tags: Optional[Sequence[str]] = resolve_param(
        value=tags, config=config, attr="wandb_tags"
    )
    _mode: Optional[Literal["online", "offline", "disabled", "shared"]] = resolve_param(
        value=mode, config=config, attr="wandb_mode"
    )

    return wandb.init(
        entity=_entity,
        project=_project,
        group=_group,
        name=_name,
        notes=_notes,
        tags=_tags,
        mode=_mode,
    )
