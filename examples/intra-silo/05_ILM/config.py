"""Configuration for the xFFL BabyLM pretraining example."""

from __future__ import annotations

import logging
import os
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
DATA_ROOT = PROJECT_ROOT / "data" / "interlinguistic-language-modeling"
MODEL_ROOT = PROJECT_ROOT / "model"
TOKENIZER_ROOT = PROJECT_ROOT / "tokenizer" / "interlinguistic-language-modeling"
OUTPUT_ROOT = PROJECT_ROOT / "output"

# -----------------------------------------------------------------------------
# Experiment constants
# -----------------------------------------------------------------------------

MODEL_NAME = os.getenv("ILM_MODEL", "BabyLM-130M")
ILM_TYPE = os.getenv("ILM_TYPE")
ILM_LANGUAGE_A = os.getenv("ILM_LANGUAGE_A")
ILM_LANGUAGE_B = os.getenv("ILM_LANGUAGE_B")
ILM_TOKENIZER = os.getenv("ILM_TOKENIZER")

if ILM_LANGUAGE_A is None or ILM_TOKENIZER is None or ILM_TYPE is None:
    logger.critical(
        f"Incomplete environment configuration: language is {ILM_LANGUAGE_A} and tokenizer is {ILM_TOKENIZER}"
    )
    raise EnvironmentError

if ILM_TYPE == "mono":
    DATASET_NAME = f"ilm_{ILM_LANGUAGE_A}"
    TOKENIZER_NAME = f"tokenizer_ilm_{ILM_TYPE}_{ILM_LANGUAGE_A}_{ILM_TOKENIZER}"
else:
    DATASET_NAME = [f"ilm_{ILM_LANGUAGE_A}", f"ilm_{ILM_LANGUAGE_B}"]
    TOKENIZER_NAME = (
        f"tokenizer_ilm_{ILM_TYPE}_{ILM_LANGUAGE_A}_{ILM_LANGUAGE_B}_{ILM_TOKENIZER}"
    )

EXPERIMENT_NAME = f"{MODEL_NAME}-{DATASET_NAME}-{TOKENIZER_NAME}"

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
            ignore_mismatched_sizes=True,
        ).resize_token_embeddings(16000)

    model: Callable = load_model


# -----------------------------------------------------------------------------
# Dataset definition
# -----------------------------------------------------------------------------


class _LazyTokenizer:
    """Singleton tokenizer loader safe for dataloader workers."""

    _tokenizer = None

    @classmethod
    def get(cls):
        if cls._tokenizer is None:
            from transformers import PreTrainedTokenizerFast

            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(TOKENIZER_ROOT / TOKENIZER_NAME / "tokenizer.json"),
                bos_token="[BOS]",
                eos_token="[EOS]",
                unk_token="[UNK]",
                pad_token="[PAD]",
            )

            cls._tokenizer = tokenizer

        return cls._tokenizer


def build_collate_fn() -> Callable:
    """Creates the dataloader collate function."""

    tokenizer = _LazyTokenizer.get()

    def collate_fn(batch: Sequence[Mapping[str, Any]]) -> Mapping[str, Tensor]:
        texts = [item["text"] if item["text"] is not None else "" for item in batch]

        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=1024,
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

    name: str | Sequence[str] = (
        DATASET_NAME
        if isinstance(DATASET_NAME, str)
        else f"{DATASET_NAME[0]}+{DATASET_NAME[1]}"
    )
    path: Path | Sequence[Path] = (
        DATA_ROOT / DATASET_NAME
        if isinstance(DATASET_NAME, str)
        else field(default_factory=lambda: [DATA_ROOT / dataset for dataset in DATASET_NAME])  # type: ignore
    )
    workers: int = 2
    batch_sizes: int = 64
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
    epochs: int = 1

    # Reproducibility
    seed: int = 42
    loglevel: int = logging.INFO

    # Output
    save_path: Path = OUTPUT_ROOT
    output_model_name: str = EXPERIMENT_NAME

    dataset_name: str = (
        DATASET_NAME
        if isinstance(DATASET_NAME, str)
        else f"{DATASET_NAME[0]}+{DATASET_NAME[1]}"
    )
    tokenizer_name: str = ILM_TOKENIZER
    ilm_type: str = ILM_TYPE

    # WandB
    wandb_params: Mapping[str, Any] = field(
        default_factory=lambda: {
            "entity": "alpha-unito",
            "project": "Interlinguistic Language Modeling - new tokenizers",
            "group": (
                DATASET_NAME
                if isinstance(DATASET_NAME, str)
                else f"{DATASET_NAME[0]}+{DATASET_NAME[1]}"
            ),
            "name": EXPERIMENT_NAME,
            "tags": [
                MODEL_NAME,
                (
                    DATASET_NAME
                    if isinstance(DATASET_NAME, str)
                    else f"{DATASET_NAME[0]}+{DATASET_NAME[1]}"
                ),
                TOKENIZER_NAME,
                ILM_TYPE,
            ],
            "mode": "online",
        }
    )
