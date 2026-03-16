"""Configuration file for the xFFL-LLM example"""

import logging
import math
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Mapping, Sequence, Type

import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from transformers import default_data_collator
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM

from xffl.custom.config import DatasetInfo, ModelInfo, XFFLConfig
from xffl.distributed.distributed_state import DistributedState
from xffl.learning.data import load_datasets_from_disk

# Constants
LLAMA3_1_8B: str = "llama3.1-8b-init"
CLEAN_MC4_IT: str = "clean_mc4_it"

BASE_PATH: str = "/leonardo_scratch/fast/uToID_bench/xffl"


@dataclass
class llama(ModelInfo):

    # LLM loading from saved model
    @staticmethod
    def _load_llm_from_checkpoint(
        config: XFFLConfig, state: DistributedState
    ) -> nn.Module:
        return LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=str(config.model_info.path),
            use_cache=True,
            local_files_only=True,  # Most HPCs do not have internet access from the nodes
            attn_implementation=config.model_info.attention,
            dtype=torch.bfloat16,  # Model is loaded in torch.bfloat16 (from the JSON file) - also "auto"
            device_map=state.init_device,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            tie_word_embeddings=True,
        )

    # Auto wrap policy
    @staticmethod
    def llama_fsdp_wrap_policy():
        return partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                LlamaDecoderLayer,
            },
        )

    name: str = LLAMA3_1_8B
    attention: str = "sdpa"  # "flash_attention_2"
    model: Callable = _load_llm_from_checkpoint
    decoder_layer: Type = LlamaDecoderLayer
    wrapping_policy: Callable = llama_fsdp_wrap_policy
    mixed_precision: MixedPrecision = field(
        default_factory=lambda: MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    )
    path: str = BASE_PATH + "/models/" + name


@dataclass
class cleanmc4it(DatasetInfo):

    @staticmethod
    def _get_cleanmc4it_splits(config: XFFLConfig, state: DistributedState):
        return load_datasets_from_disk(
            splits={"train": "train", "val": "val"},
            base_path=Path(str(config.dataset_info.path)),
        )  # Original LLaMA training packs the datasets

    name: str = CLEAN_MC4_IT
    splits: Callable = _get_cleanmc4it_splits
    batch_sizes: Mapping[str, int] = field(
        default_factory=lambda: {"train": 2, "val": 2}
    )
    subsampling: Mapping[str, int] = field(
        default_factory=lambda: {"train": 65536, "val": 4096}
    )
    workers: int = 2
    collate_fn: Callable = default_data_collator
    path: str = BASE_PATH + "/data/" + CLEAN_MC4_IT


# XFFL configuration
@dataclass
class xffl_config(XFFLConfig):

    # Optimizer
    @staticmethod
    def _get_optimizer(model: nn.Module, config: XFFLConfig) -> Optimizer:
        return AdamW(
            params=model.parameters(),
            lr=config.learning_rate,  # type: ignore
            weight_decay=config.weight_decay,  # type: ignore
            betas=config.betas,  # type: ignore
            fused=True,  # Supported only on torch.float64, torch.float32, torch.float16, and torch.bfloat16
        )

    # Default
    model_info: ModelInfo = field(default_factory=llama)
    dataset_info: DatasetInfo = field(default_factory=cleanmc4it)
    optimizer: Callable[[nn.Module, XFFLConfig], Optimizer] = _get_optimizer

    # General
    loglevel: int = logging.DEBUG
    seed: int = 42
    federated: int = 4
    federated_batches: int = 8

    # Learning
    learning_rate: float = 3e-4
    epochs: int = 1

    # WandB
    wandb_entity: str = "alpha-unito"
    wandb_project: str = "FL+DP"
    wandb_group: str = "FL+FSDP_new"
    wandb_notes: str = "EuroPar 2026 experiments"
    wandb_tags: Sequence[str] = field(default_factory=lambda: ["xFFL", "EuroPar"])
    wandb_mode: str = "offline"

    # Learning rate scheduler
    @staticmethod
    def _get_llama31_cosine_schedule(
        optimizer: Optimizer, total_steps: int, config: XFFLConfig
    ) -> LRScheduler:
        """
        Scheduler stile LLaMA3.1 semplificato: warmup -> cosine decay.

        Args:
            optimizer: torch.optim.Optimizer
            total_steps (int): passi totali (es. 128)
            lr_max (float): learning rate massimo
            warmup_frac (float): frazione di warmup (default 5%)
        """
        warmup_steps = int(total_steps * config.warmup_frac)  # type: ignore
        decay_steps = total_steps - warmup_steps

        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / max(1, warmup_steps)
            else:
                # Cosine decay
                progress = (step - warmup_steps) / max(1, decay_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step))

    # Advanced configuration
    lr_scheduler: Callable = _get_llama31_cosine_schedule

    # Custom - optimizer
    weight_decay: float = 0.1
    betas: Sequence[float] = (0.9, 0.95)
    warmup_frac: float = 0.1
