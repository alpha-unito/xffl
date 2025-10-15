"""LLM training example script

Inspired from llama-recipes' fine-tuning.py script:
https://github.com/meta-llama/llama-cookbook/blob/main/src/llama_recipes/finetuning.py
"""

import functools
import math
import os
import time
from logging import Logger, getLogger
from typing import Dict, Optional

import torch
import torch.nn as nn
import transformers
import wandb
from config import xffl_config
from datasets import Dataset, DatasetDict
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision, wrap
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel, AutoModelForCausalLM, default_data_collator

from xffl.custom.types import PathLike
from xffl.distributed import distributed
from xffl.learning import data, modelling, processing, utils
from xffl.utils.logging import setup_logging

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def pretraining(
    config: xffl_config,
    # args: argparse.Namespace, model_info: PathLike, dataset_info: PathLike
) -> None:
    """LLM pre-training script

    This is not considered fine-tuning since no parameter reduction/quantization technique is applied

    :param args: Command-line arguments
    :type args: argparse.Namespace
    :param model_info: Model path
    :type model_info: PathLike
    :param dataset_info: Dataset path
    :type dataset_info: PathLike
    """
    setup_time: float = time.perf_counter()

    # Set the requested logging level
    setup_logging(log_level=config.loglevel)

    # Sets RNGs seeds and force PyTorch's deterministic execution
    generator: Optional[torch.Generator] = (
        utils.set_deterministic_execution(seed=config.seed) if config.seed else None
    )

    # Convert paths to the container's deafults if executing inside one
    if "XFFL_IMAGE" in os.environ:
        model_path: str = str(PathLike("/model/"))
        dataset_path: str = str(PathLike("/dataset/"))
    else:
        model_path: str = str(PathLike(config.model.path + config.model.name))
        dataset_path: str = str(PathLike(config.dataset.path + config.dataset.name))

    # PyTorch's distributed backend setup
    start_time = time.perf_counter()
    state: distributed.DistributedState = distributed.setup_distributed_process_group(
        hsdp=config.hsdp,
        federated=config.federated_scaling,
        streams=config.cuda_streams,
    )
    if state.rank == 0 and torch.distributed.is_initialized():
        logger.debug(
            f"Rendez-vous time: {(time.perf_counter() - start_time):.2f} seconds"
        )

    # Large data preloading in background
    if state.node_local_rank == 0:
        utils.preload(files=[model_path])

    # WandB setup
    wandb_run: wandb.wandb_run.Run = wandb.init(  # Default entity
        entity="alpha-unito",
        project="xFFL - convergence",
        group=config.wandb_name,
        name=f"client_{state.rank}",
        notes=f"{config.model.name} pre-training on the gsarti clean_mc4_it dataset on multiple HPCs through xFFL",
        tags=["xFFL", f"{config.model.name}", "clean_mc4_it"],
        mode=config.wandb_mode,  # Set to "disable" to execute without wandb
        # config=config.__dict__,
    )

    # The whole training is done in bfloat16
    default_precision: torch.dtype = torch.bfloat16
    mixed_precision: MixedPrecision = MixedPrecision(
        param_dtype=default_precision,
        reduce_dtype=default_precision,
        buffer_dtype=default_precision,
        cast_forward_inputs=True,
    )

    # LLM loading from saved model
    start_time = time.perf_counter()
    model: AutoModel = (
        AutoModelForCausalLM.from_pretrained(  # Configuration is automatically loaded from the JSON file inside the model folder
            pretrained_model_name_or_path=model_path,  # model_info.path,
            use_cache=False,
            # output_loading_info=True #TODO: to add to verbose mode
            local_files_only=not config.online,  # Most HPCs do not have internet access from the nodes
            attn_implementation=config.attention,
            dtype=default_precision,  # Model is loaded in torch.bfloat16 (from the JSON file) - also "auto"
            device_map=state.init_device,
            use_safetensors=True,
        )
    )

    # Reset model weights
    def reset_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.reset_parameters()

    # TODO: make this an option
    if state.rank == 0:
        logger.info("Re-initializing model's weights...")
    # model.apply(reset_weights)

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
        wrapping_policy=functools.partial(
            wrap.transformer_auto_wrap_policy,
            transformer_layer_cls={
                transformers.models.llama.modeling_llama.LlamaDecoderLayer
            },
        ),
        mixed_precision=mixed_precision,
    )

    # Activation checkpointing
    # This can also be called before FSDP, will result in applying the HF-specific method, giving warnings during the training
    utils.set_activation_checkpointing(
        model=model,
        layer=config.model.decoder_layers,
    )

    if state.rank == 0:
        logger.debug(
            f"FSDP wrapping setup time: {(time.perf_counter() - start_time):.2f} seconds"
        )

    # Dataset loading
    start_time = time.perf_counter()
    datasets: Dict[str, Dataset | DatasetDict] = data.load_datasets_from_disk(
        splits=config.dataset.splits,
        base_path=dataset_path,
    )  # Original LLaMA training packs the datasets
    if state.rank == 0:
        logger.debug(
            f"Dataset loading time: {(time.perf_counter() - start_time):.2f} seconds"
        )

    # Dataloaders creation
    start_time = time.perf_counter()
    dataloaders: Dict[str, DataLoader] = {}
    for split in config.dataset.splits:

        if split == "train" and config.subsampling:
            datasets[split] = datasets[split].select(list(range(0, config.subsampling)))

        if state.rank == 0:
            logger.debug(f"{split} set size: {len(datasets[split])} samples")

        dataloaders[split] = DataLoader(
            dataset=datasets[split],
            batch_size=(
                config.train_batch_size if split == "train" else config.val_batch_size
            ),
            sampler=DistributedSampler(
                dataset=datasets[split],
                num_replicas=state.world_size,
                rank=state.rank,
                shuffle=split == "train",
                seed=config.seed if config.seed else None,
                drop_last=True,
            ),
            num_workers=config.workers,
            collate_fn=default_data_collator,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=(
                utils.seed_dataloader_worker if config.seed else None
            ),  # Necessary for reproducibility
            generator=(
                generator if config.seed else None
            ),  # Necessary for reproducibility
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
    if config.scale_learning_rate:
        config.learning_rate = config.learning_rate * state.world_size

    if state.rank == 0:
        logger.info(f"Learning rate adjusted to: {config.learning_rate}")

    # Optimizer and lr scheduler creation
    optimizer: AdamW = AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
        # foreach=True,  # Optimizes performances but uses more memory
        fused=True,  # Supported only on torch.float64, torch.float32, torch.float16, and torch.bfloat16
    )

    def get_llama31_cosine_schedule(optimizer, total_steps, warmup_frac=0.1):
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

    if config.benchmark:
        with torch.no_grad():
            if state.is_federated_scaling_setup():
                from xffl.distributed.aggregation import benchmark_aggregation

                benchmark_aggregation(
                    model=model,
                    state=state,
                    iterations=config.benchmark,
                    dump=f"{config.workspace}/{config.csv}",
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
            save_path=config.output,
            output_model_name=config.output_model,
            epochs=config.epochs,
            federated_batches=config.federated_batches,
        )

        if state.rank == 0:
            [logger.debug(f"Key: {k}, Value: {v}") for k, v in results.items()]
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
        pretraining(xffl_config)
    except KeyboardInterrupt as e:
        logger.exception(e)
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    main()
