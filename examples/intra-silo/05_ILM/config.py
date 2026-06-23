"""Configuration file for the xFFL-ILM example"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Tuple, Type

import numpy as np
import torch
from datasets import Dataset
from datasets import Dataset as HFDataset
from torch import nn
from torch.distributed.fsdp import MixedPrecision
from torch.optim import AdamW
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from xffl.custom.config import DatasetInfo, ModelInfo, OptimizerInfo, XFFLConfig
from xffl.distributed.distributed_state import DistributedState
from xffl.learning.optim import warmup_cosine_decay

# Force HuggingFace to offline mode
os.environ["HF_HUB_OFFLINE"] = "1"

# Constants
BASE_PATH: Path = Path("/beegfs/home/gmittone/xffl")

TOKENIZER_DIR = "/beegfs/home/gmittone/xffl/examples/intra-silo/05_ILM/ilm_ita"
BLOCK_SIZE = 1024  # context window (tokens)


# Model information
@dataclass
class BabyLM(ModelInfo):

    @staticmethod
    # LLM loading from saved model
    def _get_babylm_model(
        config: XFFLConfig,
        state: DistributedState,
    ) -> nn.Module:

        tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)

        VOCAB_SIZE = len(tokenizer)
        BOS_ID = tokenizer.bos_token_id
        EOS_ID = tokenizer.eos_token_id
        PAD_ID = tokenizer.pad_token_id

        llama_config = LlamaConfig(
            vocab_size=VOCAB_SIZE,  # 32 000
            # --- dimensions ---
            hidden_size=832,  # d_model (bumped from 768 to compensate for smaller embed table)
            intermediate_size=2240,  # FFN inner dim (~2.7 × hidden, SwiGLU)
            num_hidden_layers=12,  # transformer depth
            # --- attention ---
            num_attention_heads=8,  # MHA heads  (832 / 8 = 104 per head)
            num_key_value_heads=8,  # GQA: set < num_attention_heads for MQA/GQA
            # --- Llama defaults ---
            hidden_act="silu",  # SwiGLU uses SiLU gating
            max_position_embeddings=BLOCK_SIZE,  # RoPE base length — 1024
            rms_norm_eps=1e-5,
            rope_theta=10_000.0,  # RoPE base frequency (original Llama)
            # --- regularisation ---
            attention_dropout=0.0,  # Llama pre-training uses 0
            # --- special tokens ---
            bos_token_id=BOS_ID,
            eos_token_id=EOS_ID,
            pad_token_id=PAD_ID,
            # --- weight tying ---
            tie_word_embeddings=True,  # LM head shares weights with embed table
        )

        return LlamaForCausalLM(llama_config)

    name: str = "BabyLM"
    attention: str = "flash_attention_2"
    model: Callable = _get_babylm_model
    mixed_precision: MixedPrecision = field(
        default_factory=lambda: MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    )
    # activation_checkpointing: bool = True
    decoder_layer: Tuple[Type, ...] = (LlamaDecoderLayer,)


# Dataset information
@dataclass
class ILMITA(DatasetInfo):

    @staticmethod
    def _get_dataset_splits(
        config: XFFLConfig,
        state: DistributedState,
    ) -> Mapping[str, HFDataset]:

        class LlamaDataset(Dataset):
            def __init__(self, token_file: str, block_size: int, dtype=np.int32):
                self.block_size = block_size

                self.tokens = np.memmap(
                    token_file,
                    mode="r",
                    dtype=dtype,
                )

                self.num_blocks = len(self.tokens) // block_size

            def __len__(self):
                return self.num_blocks

            def __getitem__(self, idx):
                start = idx * self.block_size
                end = start + self.block_size

                ids = torch.from_numpy(self.tokens[start:end].astype(np.int64))

                return {
                    "input_ids": ids[:-1],
                    "labels": ids[1:],
                }

            def __getitems__(self, indices):
                return [self.__getitem__(idx) for idx in indices]

        return {
            "train": LlamaDataset(
                token_file="/beegfs/home/gmittone/xffl/examples/intra-silo/05_ILM/tokens.bin",
                block_size=BLOCK_SIZE,
            )
        }

    name: str = "ILM_Ita"
    splits: Callable = _get_dataset_splits
    workers: int = 4
    batch_sizes: Mapping[str, int] = field(
        default_factory=lambda: {"train": 32, "val": 32}
    )
    # subsampling: int = 1024


# Optimizer information
@dataclass
class AdamWConfig(OptimizerInfo):
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
            "warmup_fraction": 0.1,
            "final_lr_ratio": 0.1,
        }
    )
    gradient_clipping: float = 1.0
    # interleaved_optim: bool = True


# XFFL configuration
@dataclass
class xffl_config(XFFLConfig):

    # Default
    model_info: ModelInfo = field(default_factory=BabyLM)
    dataset_info: DatasetInfo = field(default_factory=ILMITA)
    optimizer_info: OptimizerInfo = field(default_factory=AdamWConfig)

    # General
    loglevel: int = logging.INFO
    seed: int = 42

    # Learning
    epochs: int = 5

    # WandB
    wandb_params: Mapping[str, Any] = field(
        default_factory=lambda: {
            "entity": "alpha-unito",
            "project": "xFFL playground",
            "group": "BabyLM - ILM ITA - pretraining - final",
            "name": "BabyLM - final",
            "notes": "Example run of BabyLM on the ILM ITA dataset",
            "tags": ["xFFL", "example", "BablyLM", "ITA"],
            "mode": "online",  # "online" to active WandB
        }
    )

    # Output
    output_folder: Optional[Path] = Path(
        "/beegfs/home/gmittone/xffl/examples/intra-silo/05_ILM/output"
    )
    output_model: Optional[str] = "BabyLM_ITA"
