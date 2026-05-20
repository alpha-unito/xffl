"""Configuration for the xFFL BabyLM pretraining example."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Type

import torch
from datasets import DatasetDict, concatenate_datasets
from torch import Tensor, nn
from torch.distributed.fsdp import MixedPrecision
from torch.optim import AdamW
from torch.utils.data import Dataset as TorchDataset

from xffl.custom.config import DatasetInfo, ModelInfo, OptimizerInfo, XFFLConfig
from xffl.distributed.distributed_state import DistributedState
from xffl.learning.data import load_datasets_from_disk
from xffl.learning.optim import warmup_cosine_decay

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""

# -----------------------------------------------------------------------------
# Paths (portable via env vars)
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path("/beegfs/home/gmittone/ILM/ILM")
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_ROOT = PROJECT_ROOT / "model"
TOKENIZER_ROOT = PROJECT_ROOT / "tokenizer" / "interlinguistic-language-modeling"
OUTPUT_ROOT = PROJECT_ROOT / "output"

# -----------------------------------------------------------------------------
# Experiment constants
# -----------------------------------------------------------------------------

MODEL_NAME = "BabyLM-130M"
DATASET = "babylm-ita"

EXPERIMENT_NAME = f"{MODEL_NAME}-{DATASET}"

# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------


@dataclass
class BabyLM(ModelInfo):
    """Model configuration for BabyLM pretraining."""

    name: str = MODEL_NAME
    attention: str = "flash_attention_2"
    path: Path = MODEL_ROOT / MODEL_NAME
    activation_checkpointing: bool = False

    decoder_layer: Type = field(init=False)  # type: ignore
    mixed_precision: MixedPrecision = field(init=False)  # type: ignore

    def __post_init__(self) -> None:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

        self.decoder_layer = Qwen3DecoderLayer
        self.mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    @staticmethod
    def load_model(config: XFFLConfig, state: DistributedState) -> nn.Module:
        """Loads the pretrained model from disk."""
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            str(config.model_info.path),
            use_cache=False,
            local_files_only=True,
            attn_implementation=config.model_info.attention,
            dtype=torch.bfloat16,
            use_safetensors=True,
        )

    model: Callable = load_model


# -----------------------------------------------------------------------------
# Dataset definition
# -----------------------------------------------------------------------------


def build_collate_fn() -> Callable:
    """Creates the dataloader collate function."""
    from transformers import AutoTokenizer

    # tokenizer = _LazyTokenizer.get()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT / MODEL_NAME)

    def collate_fn(batch: Sequence[Mapping[str, Any]]) -> Mapping[str, Tensor]:
        texts = [item["text"] if item["text"] is not None else "" for item in batch]

        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        labels = input_ids.clone()  # type: ignore
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }  # type: ignore

    return collate_fn


@dataclass
class ILMDataset(DatasetInfo):
    """Dataset configuration for ILM training."""

    @staticmethod
    def _get_splits(
        config: XFFLConfig, state: DistributedState
    ) -> Mapping[str, TorchDataset | DatasetDict]:
        """Loads dataset splits from disk."""
        dataset = None

        if isinstance(config.dataset_info.path, Path):
            dataset = load_datasets_from_disk(
                splits={"train": "train"},
                base_path=config.dataset_info.path,  # type: ignore
            )
        else:
            dataset_a = load_datasets_from_disk(
                splits={"train": "train"},
                base_path=config.dataset_info.path[0],  # type: ignore
            )
            dataset_b = load_datasets_from_disk(
                splits={"train": "train"},
                base_path=config.dataset_info.path[1],  # type: ignore
            )
            dataset = DatasetDict(
                {
                    "train": concatenate_datasets(
                        [dataset_a["train"], dataset_b["train"]]
                    ).shuffle(seed=config.seed)
                }
            )

        return dataset  # type: ignore

    name: str = DATASET
    path: Path = DATA_ROOT / DATASET
    workers: int = 2
    batch_sizes: int = 256
    # subsampling: int = 512

    splits: Callable = _get_splits
    collate_fn: Callable = field(default_factory=build_collate_fn)


# -----------------------------------------------------------------------------
# Optimizer definition
# -----------------------------------------------------------------------------


@dataclass
class AdamW(OptimizerInfo):
    """Optimizer configuration for BabyLM pretraining."""

    optimizer: Callable = AdamW

    optimizer_params: Mapping[str, Any] = field(
        default_factory=lambda: {
            "lr": 2e-3,
            "weight_decay": 0.01,
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
    gradient_clipping: float = 1.0
    # gradient_accumulation: int = 2
    # interleaved_optim: bool = True


# -----------------------------------------------------------------------------
# Main XFFL configuration
# -----------------------------------------------------------------------------


@dataclass
class BabyLMConfig(XFFLConfig):
    """Full training configuration for BabyLM pretraining."""

    # Core components
    model_info: ModelInfo = field(default_factory=BabyLM)
    dataset_info: DatasetInfo = field(default_factory=ILMDataset)
    optimizer_info: OptimizerInfo = field(default_factory=AdamW)

    # Training
    epochs: int = 10

    # Reproducibility
    seed: int = 42
    loglevel: int = logging.INFO

    # Output
    save_path: Path = OUTPUT_ROOT
    output_model_name: str = EXPERIMENT_NAME

    dataset_name: str = DATASET

    # WandB
    wandb_params: Mapping[str, Any] = field(
        default_factory=lambda: {
            "entity": "alpha-unito",
            "project": "Interlinguistic Language Modeling - new tokenizers",
            "group": DATASET,
            "name": EXPERIMENT_NAME,
            "tags": [
                MODEL_NAME,
                DATASET,
            ],
            "mode": "online",
        }
    )
