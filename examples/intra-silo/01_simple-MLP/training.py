"""Simple MLP training script"""

import time
from logging import Logger, getLogger
from typing import Mapping, MutableMapping, Optional

import torch
import torch.nn as nn
from config import xffl_config
from torch.optim import Adadelta
from torch.utils.data import DataLoader

from xffl.distributed import distributed
from xffl.learning import modelling, processing, utils
from xffl.learning.data import create_dataloaders
from xffl.utils.logging import setup_logging

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def pretraining(config: xffl_config) -> None:
    """Simple MLP training script

    :param args: Command-line arguments
    :type args: argparse.Namespace
    :param model_info: Model information class
    :type model_info: ModelInfo
    :param dataset_info: Dataset information class
    :type dataset_info: DatasetInfo
    """
    setup_time: float = time.perf_counter()

    # Set the requested logging level
    setup_logging(log_level=config.loglevel)

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

    # Model setup
    start_time: float = time.perf_counter()
    model: nn.Module = modelling.create_fsdp_model(state=state, config=config)

    # Print model's weights
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
        distributed_sampling=not config.one_class,
    )
    if state.rank == 0:
        logger.debug(
            f"Dataset loading time: {(time.perf_counter() - start_time):.2f} seconds"
        )

    # Optimizer and lr scheduler creation
    optimizer: Adadelta = Adadelta(
        params=model.parameters(),
        lr=config.learning_rate,
    )

    if state.rank == 0:
        logger.debug(
            f"Total setup time: {(time.perf_counter() - setup_time):.2f} seconds"
        )
        logger.debug(
            f"GPU RAM allocated before training: {torch.cuda.max_memory_allocated() / 10**9:.2f} GB"
        )

    # Main training function
    results: Mapping[str, float] = processing.distributed_training(
        model=model,
        state=state,
        optimizer=optimizer,
        train_dataloader=dataloaders["train"],
        val_dataloader=dataloaders["test"],
        config=config,
    )

    if state.rank == 0:
        [logger.debug(f"{k}:\t{v:.2f}") for k, v in results.items()]

    # PyTorch's distributed backend cleanup
    distributed.cleanup_distributed_process_group(
        state=state, del_obj=(model, optimizer)
    )


def main():
    """Argument parsing and training launch"""

    try:
        pretraining(xffl_config())
    except KeyboardInterrupt as e:
        logger.exception(e)
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    main()
