"""Configuration file for the xFFL-LLM+tokenizer example"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from torch import Tensor, nn
from torch.distributed.fsdp import MixedPrecision
from torch.optim import AdamW

from xffl.custom.config import DatasetInfo, ModelInfo, OptimizerInfo, XFFLConfig
from xffl.distributed.distributed_state import DistributedState
from xffl.learning.optim import warmup_cosine_decay

# Force HuggingFace to offline mode
os.environ["HF_HUB_OFFLINE"] = "1"

# Constants
BASE_PATH: Path = Path("/beegfs/home/gmittone/xffl")


def _load_fasta_dataset(
    path: Path, test_split: float = 0.2, seq_len: int = 8192, seed: int = 42
) -> DatasetDict:
    sequences: list[str] = []
    current_sequence: list[str] = []

    with open(path) as f:
        for line in f:
            line: str = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if current_sequence:
                    sequences.append("".join(current_sequence))
                    current_sequence = []
            else:
                current_sequence.append(line)

        if current_sequence:
            sequences.append("".join(current_sequence))

    chunks: list[str] = []

    # +1 per avere seq_len token dopo lo shift
    chunk_len = seq_len + 1

    for sequence in sequences:
        for start in range(0, len(sequence) - chunk_len + 1, seq_len):
            chunks.append(sequence[start : start + chunk_len])

    dataset = HFDataset.from_dict({"sequence": chunks})

    split: DatasetDict = dataset.train_test_split(
        test_size=test_split,
        seed=seed,
        shuffle=True,
    )

    return split


# Model information
@dataclass
class evo_2(ModelInfo):
    from StripedHyena2 import AttentionBlock, ParallelGatedConvBlock

    @staticmethod
    # LLM loading from saved model
    def _load_evo2_from_checkpoint(
        config: XFFLConfig, state: DistributedState
    ) -> nn.Module:
        import pkgutil

        import yaml
        from StripedHyena2 import StripedHyena
        from vortex.model.utils import dotdict

        evo2_config = yaml.safe_load(pkgutil.get_data("evo2.utils", "configs/evo2-1b-8k.yml"))  # type: ignore
        evo2_config = dotdict(evo2_config)  # type: ignore
        evo2_config.use_fp8_input_projections = False  # type: ignore

        model: nn.Module = StripedHyena(
            evo2_config, current_device=state.current_device
        )
        model.custom_load_state_dict(
            torch.load(
                str(config.model_info.path) + f"/{config.model_info.name}.pt",
                weights_only=False,
                map_location="cpu",
            ),
            strict=False,
        )

        return model.to(dtype=torch.bfloat16)

    name: str = "evo2_1b_base"
    attention: str = "flash_attention_2"
    model: Callable = _load_evo2_from_checkpoint
    mixed_precision: MixedPrecision = field(
        default_factory=lambda: MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    )
    path: Path = BASE_PATH / "model" / name


# Dataset information
@dataclass
class opengenome(DatasetInfo):

    @staticmethod
    def _get_dataset_splits(
        config: XFFLConfig, state: DistributedState
    ) -> Mapping[str, HFDataset]:
        assert state.rank is not None
        assert config.seed is not None

        dataset_dict: Mapping[int, str] = {
            0: "flye_canu.fasta",
            1: "klebsiella.fasta",
        }

        fasta_path: Path = Path(config.dataset_info.path) / dataset_dict[state.rank]  # type: ignore
        split: DatasetDict = _load_fasta_dataset(
            path=fasta_path, test_split=0.2, seq_len=2048, seed=config.seed
        )

        return {
            "train": split["train"],
            "val": split["test"],
        }

    @staticmethod
    def _get_collate_fn() -> Callable:
        from vortex.model.tokenizer import CharLevelTokenizer

        vocab_size: int = 512
        tokenizer: CharLevelTokenizer = CharLevelTokenizer(vocab_size)

        def _collate_fn(batch: Sequence[dict[str, Any]]) -> tuple[Tensor, Tensor]:

            enc = torch.tensor(
                tokenizer.tokenize_batch([sample["sequence"] for sample in batch]),
                dtype=torch.long,
            )

            inputs = enc[:, :-1]
            targets = enc[:, 1:]

            return inputs, targets

        return _collate_fn

    name: str = "genome"
    splits: Callable = _get_dataset_splits
    batch_sizes: Mapping[str, int] = field(
        default_factory=lambda: {"train": 4, "val": 1}
    )
    # subsampling: int = 64
    collate_fn: Callable = field(default_factory=_get_collate_fn)
    workers: int = 2
    path: Path = BASE_PATH / "dataset" / name


# Optimizer information
@dataclass
class AdamWConfig(OptimizerInfo):
    """Optimizer configuration for BabyLM pretraining."""

    optimizer: Callable = AdamW

    optimizer_params: Mapping[str, Any] = field(
        default_factory=lambda: {
            "lr": 1e-6,
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
    model_info: ModelInfo = field(default_factory=evo_2)
    dataset_info: DatasetInfo = field(default_factory=opengenome)
    optimizer_info: OptimizerInfo = field(default_factory=AdamWConfig)

    # General
    loglevel: int = logging.DEBUG
    seed: int = 42

    # Learning
    federated: int = 1
    federated_batches: int = 8
    epochs: int = 1

    # Custom criterion
    @staticmethod
    def _get_weighted_mse_loss(
        state: DistributedState,
    ) -> Callable[[Tensor, Tensor], Tensor]:
        def _evo2_CrossEntropy(output: Tensor, target: Tensor) -> Tensor:
            return F.cross_entropy(
                output.reshape(-1, output.size(-1)), target.reshape(-1), ignore_index=0
            )

        return _evo2_CrossEntropy

    criterion: Callable = _get_weighted_mse_loss

    # WandB
    wandb_params: Mapping[str, Any] = field(
        default_factory=lambda: {
            "entity": "alpha-unito",
            "project": "xFFL playground",
            "group": "05_Evo2",
            "name": "Example",
            "notes": "Example run of xFFL with Evo2 for bioinformatics",
            "tags": ["xFFL", "example", "Evo2"],
            "mode": "disabled",
        }
    )

    # Output
    output_folder: Optional[Path] = None
    output_model: Optional[str] = None
