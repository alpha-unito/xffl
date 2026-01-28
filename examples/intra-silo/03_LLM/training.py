"""LLM training example script

Inspired from llama-recipes' fine-tuning.py script:
https://github.com/meta-llama/llama-cookbook/blob/main/src/llama_recipes/finetuning.py
"""

import os
import time
from logging import Logger, getLogger
from typing import Any, MutableMapping, Optional

import torch
import torch.nn as nn
import wandb
from config import xffl_config
from torch.optim import AdamW
from torch.utils.data import DataLoader

from xffl.custom.config import XFFLConfig
from xffl.custom.types import PathLike
from xffl.distributed import distributed
from xffl.learning import modelling, processing, utils
from xffl.learning.data import create_dataloaders
from xffl.learning.utils import wandb_setup
from xffl.utils.logging import setup_logging

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def pretraining(config: XFFLConfig) -> None:
    """Simple MLP training script

    :param config: xFFL configuration
    :type config: XFFLConfig
    """
    setup_time: float = time.perf_counter()

    # Set the requested logging level
    setup_logging(log_level=config.loglevel)

    # Convert paths to the container's defaults if executing inside one
    if "XFFL_IMAGE" in os.environ:
        model_path: str = str(PathLike("/model/"))
        dataset_path: str = str(PathLike("/dataset/"))
    else:
        model_path: str = str(
            PathLike(str(config.model_info.path) + str(config.model_info.name))
        )
        dataset_path: str = str(
            PathLike(str(config.dataset_info.path) + str(config.dataset_info.name))
        )

    # Sets RNGs seeds and force PyTorch's deterministic execution
    generator: Optional[torch.Generator] = utils.set_deterministic_execution(
        config=config
    )

    # PyTorch's distributed backend setup
    start_time: float = time.perf_counter()
    state: distributed.DistributedState = distributed.setup_distributed_process_group(
        config=config
    )
    if state.rank == 0:
        logger.debug(
            f"Rendez-vous time: {(time.perf_counter() - start_time):.2f} seconds"
        )

    # Large data preloading in background
    # TODO: this has to improve
    if state.node_local_rank == 0:
        utils.preload(files=[model_path, dataset_path])

    # Model setup
    start_time: float = time.perf_counter()
    model: nn.Module = modelling.create_fsdp_model(state=state, config=config)

    # Activation checkpointing
    # This can also be called before FSDP, will result in applying the HF-specific method, giving warnings during the training
    if config.model_info.decoder_layers is not None:
        utils.set_activation_checkpointing(
            model=model,
            layer=config.model_info.decoder_layers,
        )

    if state.rank == 0:
        logger.debug(
            f"Model loading time: {(time.perf_counter() - start_time):.2f} seconds"
        )
        logger.debug(
            f"Training {config.model_info.name}: {(utils.get_model_size(model=model) / 1e6):.2f} million trainable parameters"
        )

    # Dataset loading
    start_time: float = time.perf_counter()
    dataloaders: Optional[MutableMapping[str, DataLoader]] = create_dataloaders(
        state=state,
        config=config,
        generator=generator,
        distributed_sampling=not config.one_class,  # type: ignore
    )
    if state.rank == 0:
        logger.debug(
            f"Dataset loading time: {(time.perf_counter() - start_time):.2f} seconds"
        )

    # Optimizer and lr scheduler creation
    optimizer: AdamW = AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,  # type: ignore
        betas=config.betas,  # type: ignore
        # foreach=True,  # Optimizes performances but uses more memory
        fused=True,  # Supported only on torch.float64, torch.float32, torch.float16, and torch.bfloat16
    )

    if state.rank == 0:
        logger.debug(
            f"Total setup time: {(time.perf_counter() - setup_time):.2f} seconds"
        )
        logger.debug(
            f"GPU RAM allocated before training: {torch.cuda.max_memory_allocated() / 10**9:.2f} GB"
        )

    # WandB setup
    wandb_run: Any = wandb_setup(config=config)

    # Main training function
    results = processing.distributed_training(
        model=model,
        state=state,
        optimizer=optimizer,
        train_dataloader=dataloaders["train"],
        val_dataloader=dataloaders["val"],
        wandb_run=wandb_run,
    )

    if state.rank == 0:
        [logger.info(f"{k}{v:.2f}") for k, v in results.items()]
        if wandb_run is not None:
            for k, v in results.items():
                wandb_run.summary[k] = v

    # PyTorch's distributed backend cleanup
    wandb.finish()
    distributed.cleanup_distributed_process_group(
        state=state, del_obj=(model, optimizer)
    )


def main():
    """Argument parsing and training launch"""

    try:
        pretraining(config=xffl_config())
    except KeyboardInterrupt as e:
        logger.exception(e)
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    main()
