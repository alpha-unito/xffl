"""Utility methods useful in a number of common DNN training scenarios"""

import functools
import os
import random
import subprocess
import sys
from logging import Logger, getLogger
from typing import List, Optional

import numpy
import torch
import torch.nn as nn
from torch import Generator
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from transformers import PreTrainedModel

from xffl.custom.config import XFFLConfig
from xffl.custom.types import PathLike
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
    model: nn.Module | PreTrainedModel, layer: type
) -> None:
    """Sets up activation (gradient) checkpointing

    This feature reduces maximum memory usage trading off more compute

    :param model: Model on which setting up the checkpointing
    :type model: nn.Module | PreTrainedModel
    :param layer: Layer to wrap, needed only by Torch models, defaults to None
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


def preload(files: List[PathLike]) -> None:
    """Pre-loads the given list of files and folders

    Particularly useful on HPC, where data can be moved near the computing nodes ahead of time

    :param files: Paths of the files and folders to be preloaded
    :type files: List[PathLike]
    :raises OSError, ValueError: If the subprocess run fails
    """
    for file in files:
        logger.debug(f"Preloading: {file}")
        command = " ".join(
            [
                "find",
                str(file),
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
