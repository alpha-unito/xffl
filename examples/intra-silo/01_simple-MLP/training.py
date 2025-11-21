"""Simple MLP training script"""

from logging import Logger, getLogger
from typing import Dict, Optional
import time

import torch
import torch.nn as nn
from torch.optim import Adadelta
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel
from torchvision import datasets, transforms

from config import xffl_config
from xffl.distributed import distributed
from xffl.learning import processing, modelling, utils
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
    generator: Optional[torch.Generator] = (
        utils.set_deterministic_execution(seed=config.seed) if config.seed else None
    )

    # PyTorch's distributed backend setup
    start_time = time.perf_counter()
    state: distributed.DistributedState = distributed.setup_distributed_process_group(
        hsdp=config.hsdp if hasattr(config, "hsdp") else None,
        federated=config.federated_scaling if hasattr(config, "federated_scaling") else None,
    )
    if state.rank == 0 and torch.distributed.is_initialized():
        logger.debug(
            f"Rendez-vous time: {(time.perf_counter() - start_time):.2f} seconds"
        )

    # Model loading from saved model
    start_time = time.perf_counter()
    model: nn.Module = config.model.class_().to(
        device=state.current_device,
        non_blocking=True,
    )

    # Print model's weights
    if state.rank == 0:
        logger.debug(
            f"Model loading time: {(time.perf_counter() - start_time):.2f} seconds"
        )
        logger.debug(
            f"Training {config.model.name}: {(utils.get_model_size(model=model) / 1e6):.2f} million trainable parameters"
        )

    # FSDP/HSDP setup
    start_time = time.perf_counter()
    model: FullyShardedDataParallel = modelling.create_fsdp_model(
        module=model,
        state=state,
    )

    if state.rank == 0:
        logger.debug(
            f"FSDP wrapping setup time: {(time.perf_counter() - start_time):.2f} seconds"
        )

    # Dataset loading
    start_time = time.perf_counter()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    xffl_datasets: Dict[str, Dataset] = {
        "train": datasets.MNIST(
            config.dataset.path,
            train=True,
            download=True,
            transform=transform,
        ),
        "test": datasets.MNIST(
            config.dataset.path,
            train=False,
            download=True,
            transform=transform,
        ),
    }

    if hasattr(config, "one_class") and config.one_class:
        for _, dataset in xffl_datasets.items():
            dataset.data = dataset.data[dataset.targets == state.rank % 10]
            dataset.targets = dataset.targets[dataset.targets == state.rank % 10]

    if hasattr(config, "subsampling"):
        xffl_datasets["train"] = Subset(xffl_datasets["train"], range(config.subsampling[0]))
        xffl_datasets["test"] = Subset(xffl_datasets["train"], range(config.subsampling[1]))

    if state.rank == 0:
        logger.debug(
            f"Dataset loading time: {(time.perf_counter() - start_time):.2f} seconds"
        )

    # Dataloaders creation
    start_time = time.perf_counter()
    dataloaders: Dict[str, DataLoader] = {}
    for split, dataset in xffl_datasets.items():

        dataloaders[split] = DataLoader(
            dataset=dataset,
            batch_size=(
                config.train_batch_size if split == "train" else config.val_batch_size
            ),
            # sampler=(
            #     DistributedSampler(
            #         dataset=dataset,
            #         num_replicas=state.world_size,
            #         rank=state.rank,
            #         shuffle=split == "train",
            #         seed=config.seed if config.seed else None,
            #         drop_last=True,
            #     )
            #     if not config.one_class
            #     else None
            # ),
            num_workers=config.workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=(
                utils.seed_dataloader_worker if config.seed else None
            ),  # Necessary for reproducibility
            generator=generator if config.seed else None,  # Necessary for reproducibility
        )

        if state.rank == 0:
            logger.debug(
                f"{split} dataloader size: {len(dataloaders[split])} mini-batches"
            )

    if state.rank == 0:
        logger.debug(
            f"Dataloaders creation time: {(time.perf_counter() - start_time):.2f} seconds"
        )

    # Optimizer and lr scheduler creation
    optimizer: Adadelta = Adadelta(
        params=model.parameters(),
        lr=config.learning_rate,
    )

    # Clear GPU cache and reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    if state.rank == 0:
        logger.debug(
            f"Total setup time: {(time.perf_counter() - setup_time):.2f} seconds"
        )
        logger.debug(
            f"GPU RAM allocated before training: {torch.cuda.max_memory_allocated() / 10**9:.2f} GB"
        )

    # Main training function
    results = processing.distributed_training(
        model=model,
        state=state,
        optimizer=optimizer,
        train_dataloader=dataloaders["train"],
        validate=True,
        eval_dataloader=dataloaders["test"],
        epochs=config.epochs,
        criterion=nn.NLLLoss(),
    )

    if state.rank == 0:
        [logger.debug(f"Key: {k}, Value: {v}") for k, v in results.items()]

    # PyTorch's distributed backend cleanup
    distributed.cleanup_distributed_process_group(
        state=state, del_obj=[model, optimizer]
    )


def main():
    """Argument parsing and training launch"""

    try:
        pretraining(xffl_config)
    except KeyboardInterrupt as e:
        logger.exception(e)
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    main()
