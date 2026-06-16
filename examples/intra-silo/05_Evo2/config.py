"""Configuration file for the xFFL-LLM+tokenizer example"""

import logging
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
from torch.optim import AdamW

from xffl.custom.config import DatasetInfo, ModelInfo, OptimizerInfo, XFFLConfig
from xffl.distributed.distributed_state import DistributedState
from xffl.learning.optim import warmup_cosine_decay

# Force HuggingFace to offline mode
os.environ["HF_HUB_OFFLINE"] = "1"

# Constants
BASE_PATH: Path = Path("/home/gmittone/xffl")


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
    decoder_layer: Tuple[Type, ...] = (AttentionBlock, ParallelGatedConvBlock)
    activation_checkpointing: bool = True
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
        from vortex.model.tokenizer import CharLevelTokenizer

        tokenizer: CharLevelTokenizer = CharLevelTokenizer(512)

        def _collate_fn(batch: Sequence[Mapping[str, Any]]) -> Tuple[Tensor, Tensor]:
            enc: Tensor = torch.tensor(
                tokenizer.tokenize_batch([sample["text"] for sample in batch]),
                dtype=torch.long,
            )

            inputs = enc[:, :-1]
            targets = enc[:, 1:]

            return inputs, targets

        return _collate_fn

    name: str = "opengenome2-metagenomes-plantcad2-c4096"
    splits: Callable = _get_dataset_splits
    batch_sizes: Mapping[str, int] = field(
        default_factory=lambda: {"train": 8, "val": 1}
    )
    subsampling: Mapping[str, int] = field(
        default_factory=lambda: {"train": 1000, "val": 20}
    )
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
    epochs: int = 1

    # Custom criterion
    @staticmethod
    def _evo2_CrossEntropy(output: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(
            output.reshape(-1, output.size(-1)), target.reshape(-1), ignore_index=0
        )

    criterion: Callable = _evo2_CrossEntropy

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
