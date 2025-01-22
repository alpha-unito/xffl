"""Utility methods useful in a number of common DNN training scenarios
"""

import random
from typing import Optional

import numpy
import torch
import torch.nn as nn
import os


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

    torch.utils.deterministic.fill_uninitialized_memory=True  # This should be True by default
    torch.use_deterministic_algorithms(mode=True)

    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8" #TODO: check cuBLAS version

    return generator


def set_nondeterministic_execution() -> None:
    """Deactivate deterministic execution and deterministic memory filling to improve performance"""
    torch.utils.deterministic.fill_uninitialized_memory=False
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


def seed_dataloader_worker() -> None:
    """Seeds PyTorch's data loader workers to guarantee reproducibility

    Since each data loader worker is a process, the underlying libraries have to be re-seeded in a reproducible way to guarantee reproducibility of the loaded data and that different workers do not have the same seed, thus loading the same data
    """
    worker_seed = (
        torch.initial_seed()
    )  # Get PyTorch generated seed for the worker (base_seed + worker_id)
    random.seed(worker_seed)
    numpy.random.seed(worker_seed)
