"""Utility methods useful in a number of common DNN training scenarios"""

import functools
import os
import random
from logging import Logger, getLogger
from typing import Optional

import numpy
import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from transformers import PreTrainedModel

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def set_deterministic_execution(seed: int) -> torch.Generator:
    """Set all the necessary RNGs to obtain reproducible executions

    This method sets random, numpy, torch and CUDA RNGs with the same seed.
    It also forces PyTorch's to use deterministic algorithms, reducing performance

    :param seed: Random seed
    :type seed: int
    :return: PyTorch RNG
    :rtype: torch.Generator
    """
    random.seed(seed)
    numpy.random.seed(seed)
    generator = torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.utils.deterministic.fill_uninitialized_memory = (
        True  # This should be True by default
    )
    torch.use_deterministic_algorithms(mode=True)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # TODO: check cuBLAS version

    return generator


def set_nondeterministic_execution() -> None:
    """Deactivate deterministic execution and deterministic memory filling to improve performance"""
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.use_deterministic_algorithms(mode=False)


def setup_gpu(rank: Optional[int] = None) -> None:
    """PyTorch GPU setup

    Sets the GPU for the current process and empties its cache
    If None defaults to "cuda"

    :param rank: Rank of the current process (local rank for multi-node trainings), defaults to None
    :type rank: Optional[int], optional
    """
    torch.cuda.set_device(rank if rank is not None else "cuda")
    torch.cuda.empty_cache()


def get_model_size(model: nn.Module) -> int:
    """Returns the model's trainable parameters number

    :param model: PyTorch model
    :type model: nn.Module
    :return: Number of trainable parameters
    :rtype: int
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    model: nn.Module | PreTrainedModel, layer: Optional[nn.Module] = None
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
        # model.enable_input_require_grads()  # TODO: Solo per finetuning?
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
