"""LLM training example script

Inspired from llama-recipes' fine-tuning.py script:
https://github.com/meta-llama/llama-cookbook/blob/main/src/llama_recipes/finetuning.py
"""

import argparse
import math
import sys
import time
from logging import Logger, getLogger
from parser import parser
from typing import Dict, Optional

import torch
import torch.nn as nn
import wandb
from datasets import Dataset, DatasetDict
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel, AutoModelForCausalLM, default_data_collator

from xffl.custom import DATASETS, MODELS, DatasetInfo, ModelInfo
from xffl.distributed import distributed
from xffl.learning import data, modelling, processing, utils
from xffl.utils.logging import setup_logging

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def pretraining(
    args: argparse.Namespace, model_info: ModelInfo, dataset_info: DatasetInfo
) -> None:
    """LLM pre-training script

    This is not considered fine-tuning since no parameter reduction/quantization technique is applied

    :param args: Command-line arguments
    :type args: argparse.Namespace
    :param model_info: Model information class
    :type model_info: ModelInfo
    :param dataset_info: Dataset information class
    :type dataset_info: DatasetInfo
    """
    setup_time: float = time.perf_counter()

    # Set the requested logging level
    setup_logging(log_level=args.loglevel)

    # Sets RNGs seeds and force PyTorch's deterministic execution
    generator: Optional[torch.Generator] = (
        utils.set_deterministic_execution(seed=args.seed) if args.seed else None
    )

    # PyTorch's distributed backend setup
    start_time = time.perf_counter()
    state: distributed.DistributedState = distributed.setup_distributed_process_group(
        hsdp=args.hsdp,
        federated=args.federated_scaling,
        # streams=args.cuda_streams,
    )
    if state.rank == 0 and torch.distributed.is_initialized():
        logger.debug(
            f"Rendez-vous time: {(time.perf_counter() - start_time):.2f} seconds"
        )

    # Large data preloading in background
    if state.node_local_rank == 0:
        utils.preload(files=[model_info.path])

    # WandB setup
    wandb_run: wandb.wandb_run.Run = wandb.init(  # Default entity
        entity="alpha-unito",
        project="xFFL - convergence",
        group=args.wandb_name,
        name=f"client_{state.rank}",
        notes=f"{args.model} pre-training on the gsarti clean_mc4_it dataset on multiple HPCs through xFFL",
        tags=["xFFL", f"{args.model}", "clean_mc4_it"],
        mode=args.wandb_mode,  # Set to "disable" to execute without wandb
        config=vars(args),
    )

    # The whole training is done in bfloat16
    default_precision: torch.dtype = torch.bfloat16
    # mixed_precision: MixedPrecision = MixedPrecision(
    #     param_dtype=default_precision,
    #     reduce_dtype=default_precision,
    #     buffer_dtype=default_precision,
    #     cast_forward_inputs=True,
    # )

    # LLM loading from saved model
    start_time = time.perf_counter()
    model: AutoModel = (
        AutoModelForCausalLM.from_pretrained(  # Configuration is automatically loaded from the JSON file inside the model folder
            pretrained_model_name_or_path=model_info.path,
            use_cache=False,
            # output_loading_info=True #TODO: to add to verbose mode
            local_files_only=not args.online,  # Most HPCs do not have internet access from the nodes
            attn_implementation=args.attention,
            dtype=default_precision,  # Model is loaded in torch.bfloat16 (from the JSON file) - also "auto"
            device_map=state.init_device,
            use_safetensors=True,
        )
    )

    # Reset model weights
    # def reset_weights(module):
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         module.reset_parameters()

    # TODO: make this an option
    # if state.rank == 0:
    #     logger.info("Re-initializing model's weights...")
    # model.apply(reset_weights)

    # Print model's weights
    if state.rank == 0:
        logger.debug(
            f"Model loading time: {(time.perf_counter() - start_time):.2f} seconds"
        )
        logger.debug(
            f"Training {args.model}: {(utils.get_model_size(model=model) / 1e6):.2f} million trainable parameters"
        )

    # FSDP/HSDP setup
    start_time = time.perf_counter()
    model: FullyShardedDataParallel = modelling.create_fsdp_model(
        module=model,
        state=state,
        model_info=model_info,
        use_orig_params=state.is_federated_scaling_setup(),
        # mixed_precision=mixed_precision,
    )  # TODO: try with origin_params

    # Activation checkpointing
    utils.set_activation_checkpointing(
        model=model, layer=model_info.decoder_layer
    )  # This can also be called before FSDP, will result in applying the HF-specific method, giving warnings during the training

    if state.rank == 0:
        logger.debug(
            f"FSDP wrapping setup time: {(time.perf_counter() - start_time):.2f} seconds"
        )

    # Dataset loading
    start_time = time.perf_counter()
    datasets: Dict[str, Dataset | DatasetDict] = data.load_datasets_from_disk(
        splits=dataset_info.splits, base_path=dataset_info.path
    )  # Original LLaMA training packs the datasets
    if state.rank == 0:
        logger.debug(
            f"Dataset loading time: {(time.perf_counter() - start_time):.2f} seconds"
        )

    # Dataloaders creation
    start_time = time.perf_counter()
    dataloaders: Dict[str, DataLoader] = {}
    for split in dataset_info.splits:

        if split == "train" and args.subsampling:
            datasets[split] = datasets[split].select(list(range(0, args.subsampling)))

        if state.rank == 0:
            logger.debug(f"{split} set size: {len(datasets[split])} samples")

        dataloaders[split] = DataLoader(
            dataset=datasets[split],
            batch_size=(
                args.train_batch_size if split == "train" else args.val_batch_size
            ),
            sampler=DistributedSampler(
                dataset=datasets[split],
                num_replicas=state.world_size,
                rank=state.rank,
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

        if state.rank == 0:
            logger.debug(
                f"{split} dataloader size: {len(dataloaders[split])} mini-batches"
            )
    if state.rank == 0:
        logger.debug(
            f"Dataloaders creation time: {(time.perf_counter() - start_time):.2f} seconds"
        )

    # Scale the learning rate in the number of model replicas (world size)
    # This is needed to keep the effective learning rate per sample constant
    # when the global batch size changes due to data parallelism
    # TODO: Investigate this more
    if args.scale_learning_rate:
        args.learning_rate = args.learning_rate * state.world_size

    if state.rank == 0:
        logger.info(f"Learning rate adjusted to: {args.learning_rate}")

    # Optimizer and lr scheduler creation
    optimizer: AdamW = AdamW(
        params=model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        # foreach=True,  # Optimizes performances but uses more memory
        fused=True,  # Supported only on torch.float64, torch.float32, torch.float16, and torch.bfloat16
    )

    def get_llama31_cosine_schedule(optimizer, total_steps, warmup_frac=0.05):
        """
        Scheduler stile LLaMA3.1 semplificato: warmup -> cosine decay.

        Args:
            optimizer: torch.optim.Optimizer
            total_steps (int): passi totali (es. 128)
            lr_max (float): learning rate massimo
            warmup_frac (float): frazione di warmup (default 5%)
        """
        warmup_steps = int(total_steps * warmup_frac)
        decay_steps = total_steps - warmup_steps

        def lr_lambda(step):
            if step < warmup_steps:
                # warmup lineare
                return step / max(1, warmup_steps)
            else:
                # decadimento coseno
                progress = (step - warmup_steps) / max(1, decay_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step))

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

    if args.benchmark:
        with torch.no_grad():
            if state.is_federated_scaling_setup():
                from xffl.distributed.aggregation import benchmark_aggregation

                benchmark_aggregation(
                    model=model,
                    state=state,
                    iterations=args.benchmark,
                    dump=f"{args.workspace}/{args.csv}",
                )
            else:
                logger.critical("Federated scaling is required for benchmarking")
                quit()
    else:
        # Main training function
        results = processing.distributed_training(
            model=model,
            state=state,
            optimizer=optimizer,
            train_dataloader=dataloaders["train"],
            validate=False,
            eval_dataloader=dataloaders["val"],
            lr_scheduler=get_llama31_cosine_schedule(
                optimizer,
                total_steps=len(dataloaders["train"]),
            ),
            wandb_run=wandb_run,
            save_path=args.output,
            output_model_name=args.output_model,
            epochs=args.epochs,
            federated_batches=args.federated_batches,
            fedopt=False,
        )

        if state.rank == 0:
            [logger.debug(f"Key: {k}, Value: {v}") for k, v in results.items()]
            if args.wandb:
                for k, v in results.items():
                    wandb_run.summary[k] = v

    # PyTorch's distributed backend cleanup
    wandb.finish()
    distributed.cleanup_distributed_process_group(
        state=state, del_obj=[model, optimizer]
    )


def main():
    """Argument parsing and training launch"""

    try:
        args = parser.parse_args(sys.argv[1:])
        pretraining(
            args=args,
            model_info=MODELS[args.model],
            dataset_info=DATASETS[args.dataset],
        )
    except KeyboardInterrupt as e:
        logger.exception(e)
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    main()
