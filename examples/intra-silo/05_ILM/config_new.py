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
from torch import nn
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
# Constants
# -----------------------------------------------------------------------------

TOKENIZER_REPO_ID = "interlinguistic-language-modeling/tokenizer_ilm_mono_it_bpe_16000"
EXPECTED_VOCAB_SIZE = 16000
UNK_TOKEN_ID = 0
PAD_TOKEN_ID = 1
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3
MASK_TOKEN_ID = 4
EXPECTED_SPECIAL_TOKEN_IDS = {
    "[UNK]": UNK_TOKEN_ID,
    "[PAD]": PAD_TOKEN_ID,
    "[BOS]": BOS_TOKEN_ID,
    "[EOS]": EOS_TOKEN_ID,
    "[MASK]": MASK_TOKEN_ID,
}
SPECIAL_TOKEN_KWARGS = {
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
    "unk_token": "[UNK]",
    "pad_token": "[PAD]",
    "mask_token": "[MASK]",
}
BLOCK_SIZE = 1024  # must match n_positions
TEXT_COLUMN = "text"


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
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(
            vocab_size=EXPECTED_VOCAB_SIZE,
            n_positions=1024,  # max sequence length
            n_embd=768,  # hidden size
            n_layer=12,  # transformer blocks
            n_head=12,  # attention heads (must divide n_embd evenly)
            n_inner=3072,  # feedforward inner dim (4 * n_embd)
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            bos_token_id=BOS_TOKEN_ID,
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
        )

        return GPT2LMHeadModel(config)

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
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        from tokenizers.processors import TemplateProcessing
        from torch.utils.data import IterableDataset
        from transformers import PreTrainedTokenizerFast

        tokenizer_file = hf_hub_download(
            repo_id=TOKENIZER_REPO_ID,
            filename="tokenizer.json",
            repo_type="model",
        )

        backend_tokenizer = Tokenizer.from_file(tokenizer_file)
        BOS_ID = backend_tokenizer.token_to_id("[BOS]")
        EOS_ID = backend_tokenizer.token_to_id("[EOS]")

        backend_tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            pair=None,
            special_tokens=[
                ("[BOS]", BOS_ID),
                ("[EOS]", EOS_ID),
            ],
        )

        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend_tokenizer,
            **SPECIAL_TOKEN_KWARGS,
        )

        VOCAB_SIZE = len(hf_tokenizer)  # should be 16000

        if VOCAB_SIZE != EXPECTED_VOCAB_SIZE:
            raise ValueError(
                f"Expected vocab size {EXPECTED_VOCAB_SIZE}, got {VOCAB_SIZE}"
            )

        for token, expected_id in EXPECTED_SPECIAL_TOKEN_IDS.items():
            actual_id = hf_tokenizer.convert_tokens_to_ids(token)
            if actual_id != expected_id:
                raise ValueError(f"Expected {token} id {expected_id}, got {actual_id}")

        print(f"Tokenizer vocab size: {VOCAB_SIZE}")
        print(f"Tokenizer file: {tokenizer_file}")

        hf_tokenizer.model_max_length = BLOCK_SIZE

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

        def token_stream(dataset):
            """
            Converts dataset into a continuous flat token stream.
            GPT-2 style: each document is wrapped as [BOS] text [EOS].
            """
            for example in dataset:
                text = example[TEXT_COLUMN]
                if not text:
                    continue

                ids = hf_tokenizer.encode(text, add_special_tokens=True)

                for tid in ids:
                    yield tid

        def pack_stream(stream, block_size=1024):
            """
            Packs a flat token stream into fixed-length blocks of
            (block_size) tokens.

            GPT-2 causal LM shift:
                input_ids = block[:-1]   (positions 0 .. block_size-2)
                labels    = block[1:]    (positions 1 .. block_size-1)

            No padding — leftover tokens at the end are discarded.
            """
            buffer = []

            for token in stream:
                buffer.append(token)

                if len(buffer) == block_size:
                    input_ids = buffer[:-1]
                    labels = buffer[1:]

                    yield {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long),
                    }

                    buffer = []

        class GPT2IterableDataset(IterableDataset):
            def __init__(self, dataset, block_size):
                self.dataset = dataset
                self.block_size = block_size
                self._length = self._compute_length()

            def _compute_length(self):
                """
                Conta i token REALI prodotti da token_stream.
                token_stream yielda 1 token alla volta → basta contarli.
                """
                return sum(
                    1 for _ in pack_stream(token_stream(self.dataset), self.block_size)
                )

            def __iter__(self):
                return pack_stream(token_stream(self.dataset), self.block_size)

            def __len__(self):
                return self._length

        from torch.utils.data import Subset

        return {"train": GPT2IterableDataset(dataset=Subset(dataset["train"], range(1000)), block_size=BLOCK_SIZE)}  # type: ignore

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
class ConfigNew(XFFLConfig):
    """Full training configuration for BabyLM pretraining."""

    # Core components
    model_info: ModelInfo = field(default_factory=BabyLM)
    dataset_info: DatasetInfo = field(default_factory=ILMDataset)
    optimizer_info: OptimizerInfo = field(default_factory=AdamW)

    # Training
    epochs: int = 1

    # Reproducibility
    seed: int = 42
    loglevel: int = logging.DEBUG

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
            "mode": "disabled",
        }
    )
