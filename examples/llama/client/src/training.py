"""LLaMA-3.1 training example script

Inspired from llama-recipes' finetuning.py script: 
https://github.com/meta-llama/llama-cookbook/blob/main/src/llama_recipes/finetuning.py
"""

import argparse
import functools
import os
import sys
import time
from logging import Logger, getLogger
from parser import parser
from typing import Dict, Union

import torch
import wandb
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
    wrap,
)
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, LlamaForCausalLM, default_data_collator
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from wandb.wandb_run import Run

from datasets import Dataset, DatasetDict
from xffl.learning import data, distributed, processing, utils
from xffl.utils.logging import setup_logging

logger: Logger = getLogger(__name__)
"""Deafult xFFL logger"""


def pretraining(args: argparse.Namespace) -> None:
    """Pre-training script for LLaMA-3.1

    This is not considered finetuning since no parameter reduction/quantization technique is applied

    :param args: Command-line arguments
    :type args: argparse.Namespace
    """

    setup_time = time.perf_counter()

    # Set the requested logging level
    setup_logging(log_level=args.loglevel)

    # The whole training is done in bfloat16
    default_precision: torch.dtype = torch.bfloat16

    # Sets RNGs seeds and force PyTorch's deterinistic execution
    generator: torch.Generator = None
    if args.seed:
        generator = utils.set_deterministic_execution(args.seed)
        logger.info(f"RNGs seed set to: {args.seed}")

    start_time = time.perf_counter()
    # PyTorch's distributed backend setup
    rank, local_rank, world_size = distributed.setup_distributed_process_group()
    if local_rank == 0:  # Large data preloading
        utils.preload([args.model])
    if torch.distributed.is_initialized():
        if rank == 0:
            logger.debug(
                f"Randez-vous time: {(time.perf_counter() - start_time):.2f} seconds"
            )
        if torch.cuda.is_available():
            utils.setup_gpu(local_rank)
            if rank == 0:
                logger.info(f"Assigned GPUs to ranks and GPUs cache cleared")
            logger.debug(
                f"Rank {rank} located on local GPU {torch.cuda.current_device()}"
            )

    # WandB setup
    wandb_run: Run = wandb.init(  # Default entity
        project="xFFL",
        group=args.wandb_name,
        name=f"client_{rank}",
        notes="LLaMA-3.1 8B pre-training on the gsarti clean_mc4_it dataset on multiple HPCs through xFFL",
        tags=["xFFL", "LLaMA", "clean_mc4_it"],
        mode=(
            args.wandb_mode if args.wandb else "disabled"
        ),  # Set to "disable" to execute without wandb
        config=args,
    )

    # LLaMA loading from saved model
    start_time = time.perf_counter()
    model_name: str = os.path.basename(args.model)
    model: LlamaForCausalLM = (
        AutoModelForCausalLM.from_pretrained(  # Configuration is automatically loaded from the JSON file inside the model folder
            pretrained_model_name_or_path=args.model,
            use_cache=False,
            # output_loading_info=True #TODO: to add to verbose mode
            local_files_only=not args.online,  # Most HPCs do not have internet access from the nodes
            attn_implementation=args.attention,
            torch_dtype=default_precision,  # Model is loaded in torch.bfloat16 (from the JSON file) - also "auto"
        )
    )

    # Activation checkpointing 2.69s/it ENTRAMBI: 3.11s/it
    # utils.set_activation_checkpointing(
    #    model=model, layer=LlamaDecoderLayer
    # )  # This can also be called aftes FSDP specifying the layer to wrap (e.g., LlamaDecoderLayer)

    # Print model's weights
    if rank == 0:
        logger.debug(
            f"Model loading time: {(time.perf_counter() - start_time):.2f} seconds"
        )
        logger.debug(
            f"Training {model_name}: {(utils.get_model_size(model=model) / 1e6):.2f} million trainable parameters"
        )

    # FSDP setup
    start_time = time.perf_counter()
    model: FullyShardedDataParallel = FullyShardedDataParallel(
        module=model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # TODO: Enable HybridShard and HSDP
        auto_wrap_policy=functools.partial(
            wrap.transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer},
        ),
        device_id=torch.cuda.current_device(),
        forward_prefetch=True,  # 2.50s/it
        limit_all_gathers=False,  # 2.50s/it
        mixed_precision=MixedPrecision(
            param_dtype=default_precision,
            reduce_dtype=default_precision,
            buffer_dtype=default_precision,
            cast_forward_inputs=True,
        ),
    )

    # Activation checkpointing 2.51s/it
    utils.set_activation_checkpointing(
        model=model, layer=LlamaDecoderLayer
    )  # This can also be called before FSDP, will result in applying the HF-specific method, giving warnings during the training

    if rank == 0:
        logger.debug(
            f"FSDP wrapping setup time: {(time.perf_counter() - start_time):.2f} seconds"
        )

    # Dataset loading
    start_time = time.perf_counter()
    datasets: Dict[str, Union[Dataset, DatasetDict]] = data.load_datasets_from_disk(
        paths={
            # todo: hardcoded paths?
            "train": os.path.join(args.dataset, "train"),
            "val": os.path.join(args.dataset, "val"),
        }
    )  # Original LLaMA training packs the datasets
    if rank == 0:
        logger.debug(
            f"Dataset loading time: {(time.perf_counter() - start_time):.2f} seconds"
        )

    # No memory pinning and non_blocking: 2.49s/it
    # Memory pinning and non_blocking: 2.49s/it

    # Dataloaders creation
    start_time = time.perf_counter()
    dataloaders: Dict[str, DataLoader] = {}
    for split in ["train", "val"]:

        # Subsampling
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
            num_workers=args.workers,  # 1: 2.47s/it #2: 2.46s/it # 4: 2.46s/it
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
        # No optimisation 2.51s/it
        # foreach=True,  # Optimizes performances but uses more memory # 2.51s/it
        # fused=True,  # Supported only on torch.float64, torch.float32, torch.float16, and torch.bfloat16 # 2.48s/it
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
        pretraining(parser.parse_args(sys.argv[1:]))
    except KeyboardInterrupt:
        logger.exception("Unexpected keyboard interrupt")


if __name__ == "__main__":
    main()
