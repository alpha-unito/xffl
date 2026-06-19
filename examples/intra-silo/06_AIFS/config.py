"""Configuration file for the xFFL-LLM+tokenizer example"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Type

import torch
from anemoi.models.layers.block import (  # GraphTransformerMapperBlock,
    TransformerProcessorBlock,
)
from datasets import Dataset as HFDataset
from setup import (
    weighted_mse_loss,
)
from torch import Tensor, nn
from torch.distributed.fsdp import MixedPrecision
from torch.optim import AdamW
from utils import (
    AnemoiContext,
    get_aifs_model,
    get_context,
    get_dataset_splits,
    get_var_node_weights,
    pre_process_hook,
)

from xffl.custom.config import DatasetInfo, ModelInfo, OptimizerInfo, XFFLConfig
from xffl.distributed.distributed_state import DistributedState
from xffl.learning.optim import warmup_cosine_decay

# Force HuggingFace to offline mode
os.environ["HF_HUB_OFFLINE"] = "1"

# Constants
BASE_PATH: Path = Path("/home/gmittone/xffl")

PRETRAINING_YAML_PATH_EU: Path = Path(
    "/beegfs/home/matteo.colombini114/Anemoi/LAM/LUCA_GIAN_pretraining_LAM.yaml"
)
PRETRAINING_YAML_PATH_US: Path = Path(
    "/beegfs/home/matteo.colombini114/Anemoi/LAM/LUCA_GIAN_pretraining_LAM_USA.yaml"
)
MODEL_YAML_PATH: Path = Path(
    "/beegfs/home/matteo.colombini114/Anemoi/LAM/model_personalized_LAM.yaml"
)


ctx_dict: Sequence[AnemoiContext] = get_context(
    pretraining_yaml=(
        PRETRAINING_YAML_PATH_EU,
        PRETRAINING_YAML_PATH_US,
        PRETRAINING_YAML_PATH_EU,  # TODO: replace with new dataset
    ),
    model_yaml=MODEL_YAML_PATH,
)


# Model information
@dataclass
class AIFS(ModelInfo):

    @staticmethod
    # LLM loading from saved model
    def _load_aifs_from_checkpoint(
        config: XFFLConfig,
        state: DistributedState,
    ) -> nn.Module:
        return get_aifs_model(ctx=ctx_dict[state.rank], state=state)  # type: ignore

    name: str = "AIFS"
    attention: str = "flash_attention_2"
    model: Callable = _load_aifs_from_checkpoint
    mixed_precision: MixedPrecision = field(
        default_factory=lambda: MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    )


# Dataset information
@dataclass
class ERA5(DatasetInfo):

    @staticmethod
    def _get_dataset_splits(
        config: XFFLConfig,
        state: DistributedState,
    ) -> Mapping[str, HFDataset]:
        return get_dataset_splits(ctx=ctx_dict[state.rank])  # type: ignore

    name: str = "ERA5"
    splits: Callable = _get_dataset_splits
    workers: int = 2
    batch_sizes: Mapping[str, int] = field(
        default_factory=lambda: {"train": 16, "val": 16}
    )
    subsampling: int = 64


# Optimizer information
@dataclass
class AdamWConfig(OptimizerInfo):
    """Optimizer configuration for BabyLM pretraining."""

    optimizer: Callable = AdamW
    optimizer_params: Mapping[str, Any] = field(
        default_factory=lambda: {
            "lr": 1e-4,
            "weight_decay": 0.1,
            "betas": (0.9, 0.95),
            "fused": True,
        }
    )
    lr_scheduler: Callable = warmup_cosine_decay
    lr_scheduler_params: Mapping[str, Any] = field(
        default_factory=lambda: {
            "warmup_fraction": 0.07,
            "final_lr_ratio": 0.03,
        }
    )
    gradient_clipping: float = 32.0


# XFFL configuration
@dataclass
class xffl_config(XFFLConfig):

    # Default
    model_info: ModelInfo = field(default_factory=AIFS)
    dataset_info: DatasetInfo = field(default_factory=ERA5)
    optimizer_info: OptimizerInfo = field(default_factory=AdamWConfig)

    # General
    loglevel: int = logging.DEBUG
    seed: int = 42

    # Learning
    federated: int = 1
    federated_batches: int = 1
    federated_layer: Tuple[Type, ...] = (TransformerProcessorBlock,)
    epochs: int = 1

    # Custom criterion
    @staticmethod
    def _get_weighted_mse_loss(
        state: DistributedState,
    ) -> Callable[[Tensor, Tensor], Tensor]:
        var_weights, node_weights = get_var_node_weights(
            ctx_dict[state.rank], state  # type: ignore
        )

        def _weighted_mse_loss(output: Tensor, target: Tensor) -> Tensor:
            return weighted_mse_loss(output, target, var_weights, node_weights)

        return _weighted_mse_loss

    criterion: Callable = _get_weighted_mse_loss

    @staticmethod
    def _pre_process_hook(
        model: nn.Module, batch: Any, state: DistributedState
    ) -> Tuple[Tensor, Tensor]:
        return pre_process_hook(model, batch, state, ctx_dict[state.rank])  # type: ignore

    pre_process_hook: Callable = _pre_process_hook

    # WandB
    wandb_params: Mapping[str, Any] = field(
        default_factory=lambda: {
            "entity": "alpha-unito",
            "project": "xFFL playground",
            "group": "Full model federation",
            "name": "Rank",
            "notes": "Example run of xFFL with AIFS for climate",
            "tags": ["xFFL", "example", "AIFS"],
            "mode": "online",  # "online" to active WandB
        }
    )

    # Output
    output_folder: Optional[Path] = None
    output_model: Optional[str] = None
