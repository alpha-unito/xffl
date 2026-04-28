"""Configuration file for the xFFL-LLM example"""

import logging
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Type

import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from transformers import default_data_collator
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM
from transformers.models.mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralForCausalLM,
)

from xffl.custom.config import DatasetInfo, ModelInfo, OptimizerInfo, XFFLConfig
from xffl.distributed.distributed_state import DistributedState
from xffl.learning.data import load_datasets_from_disk
from xffl.learning.optim import warmup_cosine_decay

# Constants
TINY_RANDOM_LLAMA_3: str = "tiny-random-llama-3"
LLAMA3_1_8B: str = "llama3.1-8b"
LLAMA3_1_70B: str = "llama3.1-70b"
MIXTRAL_8x7b_v0_1: str = "mixtral-8x7b-v0.1"
CLEAN_MC4_IT: str = "clean_mc4_it"

BASE_PATH: Path = Path("/beegfs/home/gmittone/xffl/")


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
            dtype=torch.bfloat16,  # Model is loaded in torch.bfloat16 (from the JSON file) - also "auto" # This slows down model loading
            device_map=state.init_device,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            tie_word_embeddings=True,
        )

    # Auto wrap policy
    @staticmethod
    def _fsdp_wrap_policy():
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
    wrapping_policy: Callable = _fsdp_wrap_policy
    activation_checkpointing: bool = False  # True
    mixed_precision: MixedPrecision = field(
        default_factory=lambda: MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
            # cast_forward_inputs=True,
        )
    )
    path: Path = BASE_PATH / "model" / name


@dataclass
class mixtral(ModelInfo):

    # LLM loading from saved model
    @staticmethod
    def _load_llm_from_checkpoint(
        config: XFFLConfig, state: DistributedState
    ) -> nn.Module:
        return MixtralForCausalLM.from_pretrained(
            pretrained_model_name_or_path=str(config.model_info.path),
            use_cache=True,
            local_files_only=True,  # Most HPCs do not have internet access from the nodes
            attn_implementation=config.model_info.attention,
            dtype=torch.bfloat16,  # Model is loaded in torch.bfloat16 (from the JSON file) - also "auto" # This slows down model loading
            device_map=state.init_device,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            tie_word_embeddings=True,
        )

    # Auto wrap policy
    @staticmethod
    def _fsdp_wrap_policy():
        return partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                MixtralDecoderLayer,
            },
        )

    name: str = MIXTRAL_8x7b_v0_1
    attention: str = "sdpa"  # "flash_attention_2"
    model: Callable = _load_llm_from_checkpoint
    decoder_layer: Type = MixtralDecoderLayer
    wrapping_policy: Callable = _fsdp_wrap_policy
    activation_checkpointing: bool = False  # True
    mixed_precision: MixedPrecision = field(
        default_factory=lambda: MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
            # cast_forward_inputs=True,
        )
    )
    path: Path = BASE_PATH / "model" / name


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
    subsampling: int = 1024
    workers: int = 2
    collate_fn: Callable = default_data_collator
    path: Path = BASE_PATH / "dataset" / CLEAN_MC4_IT


# Optimizer information
@dataclass
class AdamW(OptimizerInfo):
    """Optimizer configuration for BabyLM pretraining."""

    optimizer: Callable = AdamW

    optimizer_params: Mapping[str, Any] = field(
        default_factory=lambda: {
            "lr": 3e-4,
            "weight_decay": 0.1,
            "betas": (0.9, 0.95),
            "fused": True,
        }
    )
    lr_scheduler: Callable = warmup_cosine_decay
    lr_scheduler_params: Mapping[str, Any] = field(
        default_factory=lambda: {
            "warmup_fraction": 0.01,
            "final_lr_ratio": 0.01,
        }
    )
    interleaved_optim: bool = True


# XFFL configuration
@dataclass
class xffl_config(XFFLConfig):

    # Default
    model_info: ModelInfo = field(default_factory=llama)
    dataset_info: DatasetInfo = field(default_factory=cleanmc4it)
    optimizer_info: OptimizerInfo = field(default_factory=AdamW)

    # General
    loglevel: int = logging.DEBUG
    seed: int = 42
    hsdp: int = 4
    federated: int = 4
    federated_batches: int = 8

    # Learning
    epochs: int = 1

    # WandB
    wandb_params: Mapping[str, Any] = field(
        default_factory=lambda: {
            "entity": "alpha-unito",
            "project": "xFFL playground",
            "group": "03_LLM",
            "name": "Prova",
            "notes": "Example run of xFFL with a LLM",
            "tags": ["xFFL", "example", "LLM"],
            "mode": "disabled",
        }
    )
