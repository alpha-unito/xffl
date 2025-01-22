"""LLaMA-3.1 training example script

Inspired from llama-recipes' finetuning.py script: 
https://github.com/meta-llama/llama-cookbook/blob/main/src/llama_recipes/finetuning.py
"""

import argparse
import functools
from typing import Dict, Union

import torch
import wandb
from datasets import Dataset, DatasetDict
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
    wrap,
)
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, Dataset, distributed
from transformers import AutoModelForCausalLM, LlamaForCausalLM, data, models

from xffl.learning import data, distributed, processing, utils


def pretraining(args: argparse.Namespace) -> None:
    """Pre-training script for LLaMA-3.1

    This is not considered finetuning since no parameter reduction/quantization technique is applied

    :param args: Command-line arguments
    :type args: argparse.Namespace
    """
    # The whole training is done in bfloat16
    default_precision: torch.dtype = torch.bfloat16

    # Sets RNGs seeds and force PyTorch's deterinistic execution
    generator: torch.Generator = None
    if args.seed:
        generator = utils.set_deterministic_execution(
            args.seed
        )  # TODO: deterministic data loaders

    # PyTorch's distributed backend setup
    rank, local_rank, world_size = distributed.setup_distributed_process_group()
    if torch.distributed.is_initialized() and torch.cuda.is_available():
        if args.verbose and rank == 0:
            print(f"Clearing GPU cache for all ranks...")
        utils.setup_gpu(local_rank)

    # WandB setup
    wandb_run: wandb.Run = wandb.init(  # Default entity
        project="xFFL",
        group=args.wandb_name,
        name=f"client_{rank}",
        notes="LLaMA-3.1 8B pre-training on the gsarti clean_mc4_it dataset on multiple HPCs through xFFL",
        tags=["xFFL", "LLaMA", "clean_mc4_it"],
        mode=(
            args.wandb_mode if args.wandb else "disable"
        ),  # Set to "disable" to execute without wandb
        config=args,
    )

    # LLaMA loading from saved model
    model: LlamaForCausalLM = (
        AutoModelForCausalLM.from_pretrained(  # Configuration is automatically loaded from the JSON file inside the model folder
            pretrained_model_name_or_path=args.model,
            use_cache=False,
            # output_loading_info=True #TODO: to add to verbose mode
            local_files_only=True,  # Most HPCs do not have internet access from the nodes
            attn_implementation=args.attention,
            torch_dtype=default_precision,  # Model is loaded in torch.bfloat16 (from the JSON file) - also "auto"
        )
    )

    # Tokenizer loading and setup
    # tokenizer: LlamaTokenizerFast = (
    #    AutoTokenizer.from_pretrained(  # Configuration is automatically loaded from the JSON file inside the model folder
    #        pretrained_model_name_or_path=args.model,
    #        local_files_only=True,  # Most HPCs do not have internet access from the nodes
    #    )
    # )
    # if not tokenizer.pad_token_id:
    #    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Print model's weights
    if args.verbose and rank == 0:
        print(
            f"Training {args.name}: {utils.get_model_size(model=model) / 1e6} million trainable parameters"
        )

    # FSDP setup
    model: FullyShardedDataParallel[LlamaForCausalLM] = FullyShardedDataParallel(
        module=model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=functools.partial(
            wrap.transformer_auto_wrap_policy,
            transformer_layer_cls={models.llama.modeling_llama.LlamaDecoderLayer},
        ),
        device_id=torch.cuda.current_device(),
        forward_prefetch=True,
        limit_all_gathers=False,
        mixed_precision=MixedPrecision(
            param_dtype=default_precision,
            reduce_dtype=default_precision,
            buffer_dtype=default_precision,
            cast_forward_inputs=True,
        ),
    )  # Original LLaMA training adds activation checkpointing

    # Dataset loading
    datasets: Dict[str, Union[Dataset, DatasetDict]] = data.load_datasets_from_disk(
        paths={
            # todo: hardcoded paths?
            "train": os.path.join(args.dataset, "clean_mc4_it_train.hf"),
            "val": os.path.join(args.dataset, "clean_mc4_it_val.hf"),
        }
    )  # Original LLaMA training packs the datasets

    # Dataloaders creation
    dataloaders: Dict[str, DataLoader] = {}
    for split in ["train", "val"]:
        if args.verbose and rank == 0:
            print(f"{split} set size: {len(datasets[split])} samples")

        dataloaders[split] = DataLoader(
            dataset=datasets[split],
            batch_size=4 if split == "train" else 1,
            sampler=distributed.DistributedSampler(
                dataset=datasets[split],
                num_replicas=world_size,
                rank=rank,
                shuffle=split == "train",
                seed=args.seed,
                drop_last=True,
            ),
            num_workers=0,  # TODO: to be investigated; if > 0 then also prefetch_factor should be set
            collate_fn=data.DefaultDataCollator,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=utils.seed_dataloader_worker,  # Necessary for reproducibility
            generator=generator,  # Necessary for reproducibility
            pin_memory_device=local_rank,
        )

        if args.verbose and rank == 0:
            print(f"{split} dataloader size: {len(dataloaders[split])} minibatches")

    # Optimizer and lr scheduler creation
    # TODO: lots of hardcoded parameters
    optimizer: AdamW = AdamW(
        params=model.parameters(),
        lr=float(1e-4),
        weight_decay=float(0),
        foreach=True,  # Optimizes performances but uses more memory
        fused=True,  # Supported only on torch.float64, torch.float32, torch.float16, and torch.bfloat16
    )
    scheduler: lr_scheduler.StepLR = lr_scheduler.StepLR(
        optimizer=optimizer, step_size=1, gamma=0.85
    )

    # Main training function
    results = processing.train(
        model,
        dataloaders["train"],
        dataloaders["val"],
        optimizer,
        scheduler,
        local_rank,
        rank,
        wandb_run,
    )

    if args.verbose and rank == 0:
        [print(f"Key: {k}, Value: {v}") for k, v in results.items()]
        if args.wandb:
            for k, v in results.items():
                wandb_run.summary[k] = v

    # PyTorch's distributed backend cleanup
    distributed.cleanup_distributed_process_group()


def main():
    """Argument parsing and training launch"""
    parser = argparse.ArgumentParser(
        prog="Cross-Facility Federated Learning (xFFL) - LLaMA example",
        description="This xFFL example pre-trains a LLaMA-3.1 8B model on multiple HPC infrastructures.",
    )

    parser.add_argument(
        "-m", "--model", help="Path to a saved model's folder", type=str, required=True
    )

    ####################
    # todo: add args
    # wandb_name
    # wandb_mode
    ####################

    parser.add_argument(
        "-attn",
        "--attention",
        help="Type of attention implementation to use",
        type=str,
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
    )

    parser.add_argument(
        "-d", "--dataset", help="Path to the dataset's folder", type=str, required=True
    )

    parser.add_argument(
        "-s",
        "--seed",
        help="Random execution seed (for reproducibility purposes)",
        type=int,
    )

    parser.add_argument(
        "-v", "--verbose", help="Enable verbose output", type=bool, action="store_true"
    )

    parser.add_argument(
        "-w", "--wandb", help="Enable WandB", type=bool, action="store_true"
    )

    parser.add_argument(
        "-name",
        "--wandb-name",
        help="WandB group name",
        type=str,
        default="LLaMA-3.1 8B",
    )

    parser.add_argument(
        "-mode",
        "--wandb-mode",
        help="WandB mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )

    try:
        pretraining(parser.parse_args())
    except KeyboardInterrupt:
        print("Unexpected keyboard interrupt")


if __name__ == "__main__":
    main()
