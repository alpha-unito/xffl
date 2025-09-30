"""Simple ResNet training script"""

import argparse
import sys
from logging import Logger, getLogger
from parser import parser
from typing import Dict, Optional

import torch
import torch.nn as nn
import wandb
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, models, transforms

from xffl.distributed import distributed
from xffl.learning import processing, utils
from xffl.utils.logging import setup_logging

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def training(args: argparse.Namespace) -> None:
    """Resnet training script

    :param args: Command-line arguments
    :type args: argparse.Namespace
    :param model_info: Model information class
    :type model_info: ModelInfo
    :param dataset_info: Dataset information class
    :type dataset_info: DatasetInfo
    """
    # Set the requested logging level
    setup_logging(log_level=args.loglevel)

    # Sets RNGs seeds and force PyTorch's deterministic execution
    generator: Optional[torch.Generator] = (
        utils.set_deterministic_execution(seed=args.seed) if args.seed else None
    )

    # PyTorch's distributed backend setup
    state: distributed.DistributedState = distributed.setup_distributed_process_group(
        hsdp=args.hsdp, federated=args.federated_scaling, streams=args.cuda_streams
    )

    # WandB setup
    wandb_run: wandb.wandb_run.Run = wandb.init(  # Default entity
        project="xFFL",
        group=args.wandb_name,
        name=f"client_{state.rank}",
        notes="Simple MLP training on the MNIST",
        tags=["xFFL", "MLP", "MNIST"],
        mode=args.wandb_mode,  # Set to "disable" to execute without wandb
        config=vars(args),
    )

    # Model loading from saved model
    model: nn.Module = models.resnet18().to(
        device=state.current_device,
        non_blocking=True,
    )

    # Print model's weights
    if state.rank == 0:
        logger.debug(
            f"Training a simple MLP: {(utils.get_model_size(model=model) / 1e6):.4f} million trainable parameters"
        )

    # Dataset loading
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    xffl_datasets: Dict[str, Dataset] = {
        "train": datasets.CIFAR10(
            "/leonardo/pub/userexternal/gmittone/data",
            train=True,
            download=True,
            transform=transform,
        ),
        "test": datasets.CIFAR10(
            "/leonardo/pub/userexternal/gmittone/data",
            train=False,
            download=True,
            transform=transform,
        ),
    }

    if args.one_class:
        for _, dataset in xffl_datasets.items():
            class_indices = [
                i for i, (_, label) in enumerate(dataset) if label == state.rank % 10
            ]
            dataset = Subset(dataset, class_indices)

    # Dataloaders creation
    dataloaders: Dict[str, DataLoader] = {}
    for split, dataset in xffl_datasets.items():

        dataloaders[split] = DataLoader(
            dataset=dataset,
            batch_size=(
                args.train_batch_size if split == "train" else args.val_batch_size
            ),
            sampler=(
                DistributedSampler(
                    dataset=dataset,
                    num_replicas=state.world_size,
                    rank=state.rank,
                    shuffle=split == "train",
                    seed=args.seed if args.seed else None,
                    drop_last=True,
                )
                if not args.one_class
                else None
            ),
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=(
                utils.seed_dataloader_worker if args.seed else None
            ),  # Necessary for reproducibility
            generator=generator if args.seed else None,  # Necessary for reproducibility
        )

        if state.rank == 0:
            logger.debug(
                f"{split} dataloader size: {len(dataloaders[split])} mini-batches"
            )

    # Learning rate adjusting
    if state.is_federated_scaling_setup():
        args.learning_rate = (
            state.federated_local_size[state.federated_rank]
            * args.learning_rate
            / state.node_local_size
        )
    else:
        args.learning_rate = (
            state.world_size * args.learning_rate / state.node_local_size
        )

    if state.rank == 0:
        logger.debug(f"Learning rate adjusted to: {args.learning_rate}")

    # Optimizer and lr scheduler creation
    optimizer: SGD = SGD(
        params=model.parameters(), lr=args.learning_rate, momentum=args.momentum
    )

    # Main training function
    results = processing.distributed_training(
        model=model,
        state=state,
        optimizer=optimizer,
        train_dataloader=dataloaders["train"],
        eval_dataloader=dataloaders["test"],
        wandb_run=wandb_run,
        epochs=args.epochs,
        federated_batches=args.federated_batches,
        criterion=nn.CrossEntropyLoss(),
    )

    if state.rank == 0:
        [logger.debug(f"Key: {k}, Value: {v}") for k, v in results.items()]
        if args.wandb:
            for k, v in results.items():
                wandb_run.summary[k] = v

    # PyTorch's distributed backend cleanup
    wandb.finish()
    distributed.cleanup_distributed_process_group(state=state)


def main():
    """Argument parsing and training launch"""

    try:
        args = parser.parse_args(sys.argv[1:])
        training(args=args)
    except KeyboardInterrupt as e:
        logger.exception(e)
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    main()
