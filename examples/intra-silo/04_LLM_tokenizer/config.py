"""Configuration file for the xFFL-LLM+tokenizer example"""

import functools
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence, Tuple, Type

import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecision, wrap
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from xffl.custom.config import DatasetInfo, ModelInfo, XFFLConfig
from xffl.distributed.distributed_state import DistributedState
from xffl.learning.data import load_datasets_from_disk

# Constants
BABYLM: str = "ccc-italian-babylm-130m"
BABYLM_MOE: str = "ccc-italian-babylm-moe"
BABYLM_ALIBI: str = "ccc-italian-babylm-130m_alibi"

BABYLM_DATASET: str = "BabyLM_Dataset_291025/tokenized"
BABYLM_DATASET_SPECIAL_TOKENS: str = "babyLM-special-tokens/tokenized"

BASE_PATH: str = "/beegfs/home/gmittone/xffl"


def _get_babylm_cosine_schedule(
    optimizer: Optimizer, total_steps: int, config: XFFLConfig
) -> LRScheduler:

    class _WarmupCosineLREpochsAccum(LRScheduler):
        def __init__(
            self,
            optimizer=optimizer,
            epochs=config.epochs,
            accum_steps=config.gradient_accumulation,
            peak_lr=config.learning_rate,
            steps_per_epoch=total_steps,
            warmup_frac=0.01,
            final_lr_ratio=0.1,
        ):
            """
            Warmup + Cosine Decay compatible with Gradient Accumulation.
            """
            self.optimizer = optimizer

            self.peak_lr = peak_lr
            self.final_lr = peak_lr * final_lr_ratio

            # Converte tutto in "optimizer steps"
            self.effective_steps_per_epoch = steps_per_epoch // accum_steps  # type: ignore
            self.total_steps = epochs * self.effective_steps_per_epoch
            self.warmup_steps = int(self.total_steps * warmup_frac)

            self.step_num = 0  # conta gli optimizer.step()

        def step(self):
            self.step_num += 1
            lr = self.get_lr()

            for group in self.optimizer.param_groups:
                group["lr"] = lr

            return lr

        def get_lr(self):
            # Warmup lineare
            if self.step_num < self.warmup_steps:
                return self.peak_lr * (self.step_num / self.warmup_steps)

            # Cosine decay
            progress = (self.step_num - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            progress = min(max(progress, 0.0), 1.0)

            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return self.final_lr + (self.peak_lr - self.final_lr) * cosine

    return _WarmupCosineLREpochsAccum()


# Model information
@dataclass
class babylm(ModelInfo):

    @staticmethod
    # LLM loading from saved model
    def _load_babylm_from_checkpoint(
        config: XFFLConfig, state: DistributedState
    ) -> nn.Module:
        module: nn.Module
        module, _ = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=str(config.model_info.path),
            use_cache=False,
            output_loading_info=config.loglevel == logging.DEBUG,
            local_files_only=True,  # Most HPCs do not have internet access from the nodes
            attn_implementation=config.model_info.attention,
            dtype=torch.bfloat16,  # Model is loaded in torch.bfloat16 (from the JSON file) - also "auto"
            device_map=state.init_device,
            use_safetensors=True,
        )  # type: ignore
        return module

    name: str = BABYLM
    attention: str = "flash_attention_2"
    model: Callable = _load_babylm_from_checkpoint
    decoder_layer: Type = Qwen3DecoderLayer
    wrapping_policy: Callable = functools.partial(
        wrap.transformer_auto_wrap_policy,
        transformer_layer_cls={decoder_layer},
    )
    path: str = BASE_PATH + "/model/" + name


# Dataset information
@dataclass
class babylm_dataset(DatasetInfo):

    @staticmethod
    def _get_babylm_dataset_splits(config: XFFLConfig, state: DistributedState):
        return load_datasets_from_disk(
            splits={"train": "train", "val": "test"},
            base_path=Path(str(config.dataset_info.path)),
        )  # Original LLaMA training packs the datasets

    name: str = BABYLM_DATASET
    splits: Callable = _get_babylm_dataset_splits
    batch_sizes: Mapping[str, int] = field(
        default_factory=lambda: {"train": 64, "val": 1}
    )
    workers: int = 2
    path: str = BASE_PATH + "/dataset/" + name


# XFFL configuration
@dataclass
class xffl_config(XFFLConfig):

    # Default
    model_info: ModelInfo = field(default_factory=babylm)
    dataset_info: DatasetInfo = field(default_factory=babylm_dataset)

    # General
    loglevel: int = logging.DEBUG
    seed: int = 42

    # Learning
    learning_rate: float = 2e-4
    epochs: int = 1
    gradient_clipping: float = 1.0
    gradient_accumulation: int = 1

    # WandB
    wandb_entity: str = "alpha-unito"
    wandb_project: str = "xFFL playground"
    wandb_group: str = "02_CNN"
    wandb_name: str = "Example"
    wandb_notes: str = "Example run of xFFL with a CNN"
    wandb_tags: Sequence[str] = field(
        default_factory=lambda: ["xFFL", "example", "MLP"]
    )
    wandb_mode: str = "disabled"

    # Advanced configuration
    mixed_precision: MixedPrecision = field(
        default_factory=lambda: MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        )
    )
    lr_scheduler: Callable = _get_babylm_cosine_schedule

    # Custom
    final_lr_ratio: float = 0.01
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    special_tokens: Tuple[str, ...] = (
        "<WIKI>",  # Wikipedia
        "<LIB>",  # LiberLiber
        "<NEWS>",  # News
        "<CONV>",  # Conversations
        "<SUBT>",  # Subtitles
        "<TWIT>",  # Twitter
        "<FORUM>",  # Forums
        "<LAW>",  # Law texts
        "<OTHER>",  # Other
    )
