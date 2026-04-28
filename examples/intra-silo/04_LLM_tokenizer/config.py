"""Configuration file for the xFFL-LLM+tokenizer example"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Type

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset
from torch import Tensor, nn
from torch.distributed.fsdp import MixedPrecision
from torch.optim import AdamW
from torch.utils.data import Dataset as TorchDataset

from xffl.custom.config import DatasetInfo, ModelInfo, OptimizerInfo, XFFLConfig
from xffl.distributed.distributed_state import DistributedState
from xffl.learning.data import load_datasets_from_disk
from xffl.learning.optim import warmup_cosine_decay

# Force HuggingFace to offline mode
os.environ["HF_HUB_OFFLINE"] = "1"

# Constants
BABYLM: str = "ccc-italian-babylm-130m"
BABYLM_MOE: str = "ccc-italian-babylm-moe"
BABYLM_ALIBI: str = "ccc-italian-babylm-130m_alibi"

BABYLM_DATASET: str = "BabyLM_Dataset_291025"
BABYLM_DATASET_SPECIAL_TOKENS: str = "babyLM-special-tokens"
MLSUM_ES: str = "mlsum_es"

BASE_PATH: str = str(os.getcwd()) + "/xffl"


# Model information
@dataclass
class babylm(ModelInfo):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    @staticmethod
    # LLM loading from saved model
    def _load_babylm_from_checkpoint(
        config: XFFLConfig, state: DistributedState
    ) -> nn.Module:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=str(config.model_info.path),
            use_cache=False,
            local_files_only=True,  # Most HPCs do not have internet access from the nodes
            attn_implementation=config.model_info.attention,
            dtype=torch.float32,  # Model is loaded in torch.bfloat16 (from the JSON file) - also "auto"
            # device_map=state.init_device,
            use_safetensors=True,
        )

    name: str = BABYLM
    attention: str = "flash_attention_2"
    model: Callable = _load_babylm_from_checkpoint
    decoder_layer: Type = Qwen3DecoderLayer
    activation_checkpointing: bool = False
    mixed_precision: MixedPrecision = field(
        default_factory=lambda: MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
            # cast_forward_inputs=True,
        )
    )
    path: str = BASE_PATH + "/model/" + name


# Dataset information
@dataclass
class babylm_dataset(DatasetInfo):

    @staticmethod
    def _get_tokenized_ita_dataset_splits(
        config: XFFLConfig, state: DistributedState
    ) -> Mapping[str, TorchDataset | DatasetDict]:
        return load_datasets_from_disk(
            splits={"train": "train", "val": "test"},
            base_path=Path(str(config.dataset_info.path) + "/tokenized"),
        )

    @staticmethod
    def _get_ita_dataset_splits(
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
                split="test",
            ),
        }  # type: ignore

    @staticmethod
    def _get_esp_dataset_splits(
        config: XFFLConfig, state: DistributedState
    ) -> Mapping[str, HFDataset]:
        return load_datasets_from_disk(
            splits={"train": "train", "val": "validation"},
            base_path=Path(str(config.dataset_info.path)),
        )  # type: ignore

    @staticmethod
    def _get_collate_fn() -> Callable:
        from transformers import AutoTokenizer

        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=BASE_PATH + "/model/" + BABYLM,
            local_files_only=True,
            use_safetensors=True,
            additional_special_tokens=(
                "<WIKI>",  # Wikipedia
                "<LIB>",  # LiberLiber
                "<NEWS>",  # News
                "<CONV>",  # Conversations
                "<SUBT>",  # Subtitles
                "<TWIT>",  # Twitter
                "<FORUM>",  # Forums
                "<LAW>",  # Law texts
                "<OTHER>",  # Other
            ),
        )

        # tokenizer.pad_token = tokenizer.eos_token
        def _collate_fn(batch: Sequence[Mapping[str, Any]]) -> Tensor:
            enc: Tensor = tokenizer([item["text"] for item in batch], truncation=True, padding="max_length", return_tensors="pt")  # type: ignore
            enc["labels"] = enc["input_ids"].clone()  # type: ignore
            return enc

        return _collate_fn

    name: str = BABYLM_DATASET
    splits: Callable = _get_tokenized_ita_dataset_splits
    batch_sizes: Mapping[str, int] = field(
        default_factory=lambda: {"train": 32, "val": 64}
    )
    subsampling: Mapping[str, int] = field(
        default_factory=lambda: {"train": 80000, "val": 40000}
    )
    collate_fn: Callable = field(default_factory=_get_collate_fn)
    workers: int = 2
    path: str = BASE_PATH + "/dataset/" + BABYLM_DATASET


# Optimizer information
@dataclass
class AdamW(OptimizerInfo):
    """Optimizer configuration for BabyLM pretraining."""

    optimizer: Callable = AdamW

    optimizer_params: Mapping[str, Any] = field(
        default_factory=lambda: {
            "lr": 2e-4,
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
    gradient_clipping: float = 1.0
    gradient_accumulation: int = 1


# XFFL configuration
@dataclass
class xffl_config(XFFLConfig):

    # Default
    model_info: ModelInfo = field(default_factory=babylm)
    dataset_info: DatasetInfo = field(default_factory=babylm_dataset)
    optimizer_info: OptimizerInfo = field(default_factory=AdamW)

    # General
    loglevel: int = logging.DEBUG
    seed: int = 42

    # Learning
    epochs: int = 1

    # WandB
    wandb_params: Mapping[str, Any] = field(
        default_factory=lambda: {
            "entity": "alpha-unito",
            "project": "xFFL playground",
            "group": "04_LLM_Tokenizer",
            "name": "Example",
            "notes": "Example run of xFFL with an LLM and tokenization on-the-fly",
            "tags": ["xFFL", "example", "LLM", "tokenizer"],
            "mode": "disabled",
        }
    )

    # Output
    output_folder: Optional[Path] = None
    output_model: Optional[str] = None
