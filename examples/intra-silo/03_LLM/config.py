"""Configuration file for the xFFL-LLM example"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence, Type

import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecision
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from transformers import AutoModelForCausalLM, default_data_collator
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer

from xffl.custom.config import DatasetInfo, ModelInfo, XFFLConfig
from xffl.distributed.distributed_state import DistributedState
from xffl.learning.data import load_datasets_from_disk

# Constants
TINY_RANDOM_LLAMA_3: str = "tiny_random_Llama-3"
LLAMA3_1_8B: str = "llama3.1-8b"
LLAMA3_1_70B: str = "llama3.1-70b"
MIXTRAL_8x7b_v0_1: str = "mixtral-8x7b-v0.1"
CLEAN_MC4_IT: str = "clean_mc4_it"

BASE_PATH: str = "/beegfs/home/gmittone/xffl"


# LLM loading from saved model
def _load_llm_from_checkpoint(config: XFFLConfig, state: DistributedState) -> nn.Module:
    return AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(config.model_info.path),
        use_cache=False,
        local_files_only=True,  # Most HPCs do not have internet access from the nodes
        attn_implementation=config.model_info.attention,
        dtype=torch.bfloat16,  # Model is loaded in torch.bfloat16 (from the JSON file) - also "auto"
        device_map=state.init_device,
        use_safetensors=True,
    )


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
            # warmup lineare
            return step / max(1, warmup_steps)
        else:
            # decadimento coseno
            progress = (step - warmup_steps) / max(1, decay_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step))


@dataclass
class llama(ModelInfo):
    name: str = TINY_RANDOM_LLAMA_3
    attention: str = "sdpa"
    model: Callable = _load_llm_from_checkpoint
    decoder_layer: Type = LlamaDecoderLayer
    activation_checkpointing: bool = True
    path: str = BASE_PATH + "/model/" + name


@dataclass
class mixtral(ModelInfo):
    name: str = MIXTRAL_8x7b_v0_1
    attention: str = "sdpa"
    model: Callable = _load_llm_from_checkpoint
    decoder_layer: Type = MixtralDecoderLayer
    path: str = BASE_PATH + "/model/" + name


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
        default_factory=lambda: {"train": 4, "val": 1}
    )
    workers: int = 2
    collate_fn: Callable = default_data_collator
    path: str = BASE_PATH + "/dataset/" + CLEAN_MC4_IT


# XFFL configuration
@dataclass
class xffl_config(XFFLConfig):

    # Default
    model_info: ModelInfo = field(default_factory=llama)
    dataset_info: DatasetInfo = field(default_factory=cleanmc4it)

    # General
    loglevel: int = logging.DEBUG
    seed: int = 42

    # Learning
    learning_rate: float = 3e-4
    epochs: int = 1

    # WandB
    wandb_entity: str = "alpha-unito"
    wandb_project: str = "xFFL playground"
    wandb_group: str = "02_CNN"
    wandb_name: str = "Example"
    wandb_notes: str = "Example run of xFFL with a CNN"
    wandb_tags: Sequence[str] = field(
        default_factory=lambda: ["xFFL", "example", "MLP"]
    )
    wandb_mode: str = "online"

    # Advanced configuration
    mixed_precision: MixedPrecision = field(
        default_factory=lambda: MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        )
    )
    lr_scheduler: Callable = _get_llama31_cosine_schedule

    # Custom - optimizer
    weight_decay: float = 0.1
    betas: Sequence[float] = (0.9, 0.95)
    warmup_frac: float = 0.1
