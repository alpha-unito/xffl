"""LLM training example script

Inspired from llama-recipes' finetuning.py script: 
https://github.com/meta-llama/llama-cookbook/blob/main/src/llama_recipes/finetuning.py
"""

import argparse
import sys
import time
from logging import Logger, getLogger
from parser import parser
from typing import Dict, Union

import torch
import torch._dynamo
import wandb
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
)
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, default_data_collator
from wandb.wandb_run import Run

from datasets import Dataset, DatasetDict
from xffl.custom import DATASETS, MODELS
from xffl.learning import data, distributed, processing, utils
from xffl.utils.logging import setup_logging

logger: Logger = getLogger(__name__)
"""Deafult xFFL logger"""


def pretraining(args: argparse.Namespace, model_info, dataset_info) -> None:
    """LLM pre-training script

    This is not considered finetuning since no parameter reduction/quantization technique is applied

    :param args: Command-line arguments
    :type args: argparse.Namespace
    """
    setup_time = time.perf_counter()

    # Set the requested logging level
    setup_logging(log_level=args.loglevel)

    # Sets RNGs seeds and force PyTorch's deterinistic execution
    generator: torch.Generator = None
    if args.seed:
        generator = utils.set_deterministic_execution(seed=args.seed)

    # PyTorch's distributed backend setup
    start_time = time.perf_counter()
    rank, local_rank, world_size, device_mesh = (
        distributed.setup_distributed_process_group(hsdp=args.hsdp)
    )
    if rank == 0 and torch.distributed.is_initialized():
        logger.debug(
            f"Randez-vous time: {(time.perf_counter() - start_time):.2f} seconds"
        )

    # Large data preloading in background
    if local_rank == 0:
        utils.preload(files=[model_info.path])

    # Devices setup
    if torch.cuda.is_available():
        current_device, init_device = utils.setup_devices(
            rank=rank, local_rank=local_rank
        )

    # WandB setup
    wandb_run: Run = wandb.init(  # Default entity
        project="xFFL",
        group=args.wandb_name,
        name=f"client_{rank}",
        notes=f"{args.model_name} pre-training on the gsarti clean_mc4_it dataset on multiple HPCs through xFFL",
        tags=["xFFL", f"{args.model_name}", "clean_mc4_it"],
        mode=(
            args.wandb_mode if args.wandb else "disabled"
        ),  # Set to "disable" to execute without wandb
        config=args,
    )

    # The whole training is done in bfloat16
    default_precision: torch.dtype = torch.bfloat16

    # LLM loading from saved model
    start_time = time.perf_counter()
    model: AutoModelForCausalLM = (
        AutoModelForCausalLM.from_pretrained(  # Configuration is automatically loaded from the JSON file inside the model folder
            pretrained_model_name_or_path=model_info.path,
            use_cache=False,
            # output_loading_info=True #TODO: to add to verbose mode
            local_files_only=not args.online,  # Most HPCs do not have internet access from the nodes
            attn_implementation=args.attention,
            torch_dtype=default_precision,  # Model is loaded in torch.bfloat16 (from the JSON file) - also "auto"
            device_map=init_device,
            use_safetensors=True,
        )
    )

    # Print model's weights
    if rank == 0:
        logger.debug(
            f"Model loading time: {(time.perf_counter() - start_time):.2f} seconds"
        )
        logger.debug(
            f"Training {args.model_name}: {(utils.get_model_size(model=model) / 1e6):.2f} million trainable parameters"
        )
        if device_mesh:
            logger.debug(f"Activating HYBRID_SHARD strategy")
        else:
            logger.debug(f"Activating FULL_SHARD sharding strategy")

    # FSDP/HSDP setup
    start_time = time.perf_counter()
    model: FullyShardedDataParallel = FullyShardedDataParallel(
        module=model,
        sharding_strategy=(
            ShardingStrategy.HYBRID_SHARD
            if device_mesh
            else ShardingStrategy.FULL_SHARD
        ),
        auto_wrap_policy=model_info.wrapping_policy,
        device_id=current_device,
        forward_prefetch=True,
        limit_all_gathers=False,
        mixed_precision=MixedPrecision(
            param_dtype=default_precision,
            reduce_dtype=default_precision,
            buffer_dtype=default_precision,
            cast_forward_inputs=True,
        ),
        sync_module_states=True,
        param_init_fn=lambda module: module.to_empty(
            device=current_device, recurse=False
        ),
        device_mesh=device_mesh,
    )

    # Activation checkpointing
    utils.set_activation_checkpointing(
        model=model, layer=model_info.decoder_layer
    )  # This can also be called before FSDP, will result in applying the HF-specific method, giving warnings during the training

    if rank == 0:
        logger.debug(
            f"FSDP wrapping setup time: {(time.perf_counter() - start_time):.2f} seconds"
        )

    # Dataset loading
    start_time = time.perf_counter()
    datasets: Dict[str, Union[Dataset, DatasetDict]] = data.load_datasets_from_disk(
        splits=dataset_info.splits, base_path=dataset_info.path
    )  # Original LLaMA training packs the datasets
    if rank == 0:
        logger.debug(
            f"Dataset loading time: {(time.perf_counter() - start_time):.2f} seconds"
        )

    # Dataloaders creation
    start_time = time.perf_counter()
    dataloaders: Dict[str, DataLoader] = {}
    for split in dataset_info.splits:

        if args.subsampling:
            datasets[split] = datasets[split].select(list(range(0, args.subsampling)))

        if rank == 0:
            logger.debug(f"{split} set size: {len(datasets[split])} samples")

        dataloaders[split] = DataLoader(
            dataset=datasets[split],
            batch_size=(
                args.train_batch_size if split == "train" else args.val_batch_size
            ),
            sampler=DistributedSampler(
                dataset=datasets[split],
                num_replicas=world_size,
                rank=rank,
                shuffle=split == "train",
                seed=args.seed if args.seed else None,
                drop_last=True,
            ),
            num_workers=args.workers,
            collate_fn=default_data_collator,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=(
                utils.seed_dataloader_worker if args.seed else None
            ),  # Necessary for reproducibility
            generator=generator if args.seed else None,  # Necessary for reproducibility
        )

        if rank == 0:
            logger.debug(
                f"{split} dataloader size: {len(dataloaders[split])} minibatches"
            )
    if rank == 0:
        logger.debug(
            f"Dataloaders creation time: {(time.perf_counter() - start_time):.2f} seconds"
        )

    # Optimizer and lr scheduler creation
    optimizer: AdamW = AdamW(
        params=model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        # No optimisation
        # foreach=True,  # Optimizes performances but uses more memory
        # fused=True,  # Supported only on torch.float64, torch.float32, torch.float16, and torch.bfloat16
    )
    scheduler: lr_scheduler.StepLR = lr_scheduler.StepLR(
        optimizer=optimizer, step_size=args.step_size, gamma=args.gamma
    )

    if rank == 0:
        logger.debug(
            f"Total setup time: {(time.perf_counter() - setup_time):.2f} seconds"
        )

    # Main training function
    results = processing.fsdp_training(
        model=model,
        optimizer=optimizer,
        train_dataloader=dataloaders["train"],
        eval_dataloader=dataloaders["val"],
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        lr_scheduler=scheduler,
        wandb_run=wandb_run,
        save_path=args.output,
        output_model_name=args.output_model,
        epochs=args.epochs,
    )

    if rank == 0:
        [logger.debug(f"Key: {k}, Value: {v}") for k, v in results.items()]
        if args.wandb:
            for k, v in results.items():
                wandb_run.summary[k] = v

    # PyTorch's distributed backend cleanup
    distributed.cleanup_distributed_process_group()


def main():
    """Argument parsing and training launch"""

    try:
        args = parser.parse_args(sys.argv[1:])
        pretraining(
            args=args,
            model_info=MODELS[args.model_name](path=args.model_path),
            dataset_info=DATASETS[args.dataset_name](path=args.dataset_path),
        )
    except KeyboardInterrupt as e:
        logger.exception(e)
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    main()
