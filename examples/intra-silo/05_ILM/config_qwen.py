"""Configuration for the xFFL BabyLM pretraining example."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Type

import torch
from datasets import DatasetDict
from torch import nn
from torch.distributed.fsdp import MixedPrecision
from torch.optim import AdamW
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer

from xffl.custom.config import DatasetInfo, ModelInfo, OptimizerInfo, XFFLConfig
from xffl.distributed.distributed_state import DistributedState
from xffl.learning.data import load_datasets_from_disk

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

MODEL_NAME = "QWEN"
ILM_TYPE = "mono"
ILM_LANGUAGE_A = "it"
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
# Constants
# -----------------------------------------------------------------------------

BLOCK_SIZE = 1024
TEXT_COLUMN = "text"
DEVICE = "cuda"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


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

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map=DEVICE,
        )
        model.train()

        model.config.max_position_embeddings = BLOCK_SIZE

        return model

    model: Callable = load_model


# -----------------------------------------------------------------------------
# Dataset definition
# -----------------------------------------------------------------------------


@dataclass
class ILMDataset(DatasetInfo):
    """Dataset configuration for ILM training."""

    @staticmethod
    def _get_splits(
        config: XFFLConfig, state: DistributedState
    ) -> Mapping[str, TorchDataset | DatasetDict]:
        """Loads dataset splits from disk."""
        from torch.utils.data import IterableDataset

        dataset = load_datasets_from_disk(
            splits={"train": "train"},
            base_path=config.dataset_info.path,  # type: ignore
        )

        def token_stream(dataset):
            for ex in dataset:
                text = ex[TEXT_COLUMN]
                if not text:
                    continue
                ids = config.tokenizer(text, add_special_tokens=False).input_ids
                for t in ids:
                    yield t
                yield config.tokenizer.eos_token_id

        def pack(stream):
            buf = []
            for t in stream:
                buf.append(t)
                if len(buf) == BLOCK_SIZE:
                    yield {
                        "input_ids": torch.tensor(buf[:-1], dtype=torch.long),
                        "labels": torch.tensor(buf[1:], dtype=torch.long),
                    }
                    buf = []

        class PackedDataset(IterableDataset):
            def __init__(self, dataset):
                self.dataset = dataset

            def __iter__(self):
                return pack(token_stream(self.dataset))

            def __len__(self):
                return 1

        return {"train": PackedDataset(dataset["train"])}  # type: ignore

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
    workers: int = 0
    batch_sizes: int = 16
    # subsampling: int = 512

    splits: Callable = _get_splits


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
    # lr_scheduler: Callable = warmup_cosine_decay
    # lr_scheduler_params: Mapping[str, Any] = field(
    #     default_factory=lambda: {
    #         "warmup_fraction": 0.01,
    #         "final_lr_ratio": 0.01,
    #     }
    # )
    # gradient_clipping: float = 1.0
    # gradient_accumulation: int = 2
    # interleaved_optim: bool = True


# -----------------------------------------------------------------------------
# Main XFFL configuration
# -----------------------------------------------------------------------------


@dataclass
class Config(XFFLConfig):
    """Full training configuration for BabyLM pretraining."""

    # Core components
    model_info: ModelInfo = field(default_factory=BabyLM)
    dataset_info: DatasetInfo = field(default_factory=ILMDataset)
    optimizer_info: OptimizerInfo = field(default_factory=AdamW)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

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
            "name": EXPERIMENT_NAME + "_new_code",
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
