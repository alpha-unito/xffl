"""Configuration file for the xFFL-LLM+tokenizer example"""

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Type

import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch import Tensor, nn
from torch.distributed.fsdp import MixedPrecision
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler

from xffl.custom.config import DatasetInfo, ModelInfo, XFFLConfig
from xffl.distributed.distributed_state import DistributedState

# Force HuggingFace to offline mode
os.environ["HF_HUB_OFFLINE"] = "1"

# Constants
EVO2_1B: str = "evo2_1b_base"
OPENGENOME2: str = "opengenome2-metagenomes-plantcad2-c4096"
BASE_PATH: str = "/leonardo_scratch/fast/uToID_bench/xffl"


# Model information
@dataclass
class evo_2(ModelInfo):
    from StripedHyena2 import AttentionBlock

    @staticmethod
    # LLM loading from saved model
    def _load_from_checkpoint(config: XFFLConfig, state: DistributedState) -> nn.Module:
        import pkgutil

        import yaml
        from StripedHyena2 import StripedHyena
        from vortex.model.utils import dotdict

        config = yaml.safe_load(pkgutil.get_data("evo2.utils", "configs/evo2-1b-8k.yml"))  # type: ignore
        config = dotdict(config)  # type: ignore
        config.use_fp8_input_projections = False  # type: ignore

        return StripedHyena(config, current_device=state.current_device).to(
            dtype=torch.bfloat16
        )

    # Auto wrap policy
    @staticmethod
    def llama_fsdp_wrap_policy():
        from functools import partial

        from StripedHyena2 import AttentionBlock
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        return partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                AttentionBlock,
            },
        )

    name: str = EVO2_1B
    attention: str = "flash_attention_2"
    model: Callable = _load_from_checkpoint
    # decoder_layer: Type = AttentionBlock
    # wrapping_policy: Callable = llama_fsdp_wrap_policy
    activation_checkpointing: bool = False
    mixed_precision: MixedPrecision = field(
        default_factory=lambda: MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    )
    path: str = BASE_PATH + "/models/" + name


# Dataset information
@dataclass
class opengenome(DatasetInfo):

    @staticmethod
    def _get_dataset_splits(
        config: XFFLConfig, state: DistributedState
    ) -> Mapping[str, HFDataset]:
        return {
            "train": load_dataset(
                "parquet",
                data_dir=str(config.dataset_info.path) + "/data",
                split="train",
            ),
            "val": load_dataset(
                "parquet",
                data_dir=str(config.dataset_info.path) + "/data",
                split="validation",
            ),
        }  # type: ignore

    @staticmethod
    def _get_collate_fn() -> Callable:
        from torch.nn.utils.rnn import pad_sequence
        from vortex.model.tokenizer import CharLevelTokenizer

        tokenizer = CharLevelTokenizer(512)

        def _collate_fn(batch: Sequence[Mapping[str, str]]) -> Tuple[Tensor, Tensor]:
            tokens = torch.tensor(
                tokenizer.tokenize_batch([sample["text"] for sample in batch]),
                dtype=int,  # type: ignore
            )

            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            return inputs, targets

        return _collate_fn

    name: str = OPENGENOME2
    splits: Callable = _get_dataset_splits
    batch_sizes: Mapping[str, int] = field(
        default_factory=lambda: {"train": 2, "val": 1}
    )
    subsampling: Mapping[str, int] = field(
        default_factory=lambda: {"train": 1000, "val": 20}
    )
    collate_fn: Callable = field(default_factory=_get_collate_fn)
    workers: int = 2
    path: str = BASE_PATH + "/data/" + OPENGENOME2


# Optimizer
def _get_optimizer(model: nn.Module, config: XFFLConfig) -> Optimizer:
    return AdamW(
        params=model.parameters(),
        lr=config.learning_rate,  # type: ignore
        weight_decay=config.weight_decay,  # type: ignore
        betas=config.betas,  # type: ignore
        fused=True,
    )


# XFFL configuration
@dataclass
class xffl_config(XFFLConfig):

    # Default
    model_info: ModelInfo = field(default_factory=evo_2)
    dataset_info: DatasetInfo = field(default_factory=opengenome)
    optimizer: Callable[[nn.Module, XFFLConfig], Optimizer] = _get_optimizer

    # General
    loglevel: int = logging.DEBUG
    seed: int = 42

    # Learning
    learning_rate: float = 1e-6
    epochs: int = 1
    gradient_clipping: float = 1.0
    gradient_accumulation: int = 1

    # Custom criterion
    @staticmethod
    def _evo2_CrossEntropy(output: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(
            output.reshape(-1, output.size(-1)), target.reshape(-1), ignore_index=0
        )

    criterion: Callable = _evo2_CrossEntropy

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
    @staticmethod
    def _get_cosine_schedule(
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
                self.final_lr = peak_lr * final_lr_ratio  # type: ignore

                # Converte tutto in "optimizer steps"
                self.effective_steps_per_epoch = steps_per_epoch // accum_steps  # type: ignore
                self.total_steps = epochs * self.effective_steps_per_epoch  # type: ignore
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
                    return self.peak_lr * (self.step_num / self.warmup_steps)  # type: ignore

                # Cosine decay
                progress = (self.step_num - self.warmup_steps) / (
                    self.total_steps - self.warmup_steps
                )
                progress = min(max(progress, 0.0), 1.0)

                cosine = 0.5 * (1 + math.cos(math.pi * progress))
                return self.final_lr + (self.peak_lr - self.final_lr) * cosine  # type: ignore

        return _WarmupCosineLREpochsAccum()

    lr_scheduler: Callable = _get_cosine_schedule

    # Output
    output_folder: Optional[Path] = None
    output_model: Optional[str] = None

    # Custom
    final_lr_ratio: float = 0.01
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
