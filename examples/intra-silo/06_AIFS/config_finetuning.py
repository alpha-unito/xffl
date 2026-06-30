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
from torch import Tensor, nn
from torch.distributed.fsdp import MixedPrecision
from torch.optim import AdamW
from torch.utils.data import DataLoader
from utils import (  # post_rollout_hook,
    AnemoiContext,
    finetuning_pre_process_hook,
    get_aifs_model,
    get_context,
    get_dataset_splits,
    get_var_node_weights,
    weighted_mse_loss,
)

from xffl.custom.config import DatasetInfo, ModelInfo, OptimizerInfo, XFFLConfig
from xffl.distributed.distributed_state import DistributedState
from xffl.learning.data import create_dataloaders
from xffl.learning.optim import warmup_cosine_decay

# Force HuggingFace to offline mode
os.environ["HF_HUB_OFFLINE"] = "1"

# Constants
BASE_PATH: Path = Path("/home/gmittone/xffl")

YAML_PATH_CL1: Path = Path(
    "/beegfs/home/matteo.colombini114/Anemoi/LAM/Client1/Exp1/finetuning_LAM_exp1.yaml"
)
YAML_PATH_CL2: Path = Path(
    "/beegfs/home/matteo.colombini114/Anemoi/LAM/Client2/Exp1/finetuning_LAM_exp1.yaml"
)
YAML_PATH_CL5: Path = Path(
    "/beegfs/home/matteo.colombini114/Anemoi/LAM/Client5/Exp1/finetuning_LAM_exp1.yaml"
)

MODEL_GLOBAL_YAML_PATH: Path = Path(
    "/beegfs/home/matteo.colombini114/Anemoi/LAM/model_global_LAM.yaml"
)
# Modello per FL personalized, con +8 learnable features
MODEL_PERSONALIZED_YAML_PATH: Path = Path(
    "/beegfs/home/matteo.colombini114/Anemoi/LAM/model_personalized_LAM.yaml"
)


ctx_dict: Sequence[AnemoiContext] = get_context(
    pretraining_yaml=(
        YAML_PATH_CL1,
        YAML_PATH_CL2,
        YAML_PATH_CL5,
    ),
    model_yaml=MODEL_GLOBAL_YAML_PATH,
    finetuning=True,
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
        ctx: AnemoiContext = ctx_dict[state.rank]  # type: ignore

        model = get_aifs_model(ctx=ctx, state=state)

        if ctx.config.files.pretrained_checkpoint is not None:
            ckpt = torch.load(
                ctx.config.files.pretrained_checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            param_keys = {name for name, _ in model.named_parameters()}
            compatible = {k: v for k, v in ckpt.items() if k in param_keys}
            model.load_state_dict(compatible, strict=False)

        return model

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
        splits, config.optimizer_info.total_steps_per_epoch = get_dataset_splits(ctx=ctx_dict[state.rank], batch_size=config.dataset_info.batch_sizes)  # type: ignore
        return splits  # type: ignore

    name: str = "ERA5"
    splits: Callable = _get_dataset_splits
    workers: int = 2
    batch_sizes: int = 16
    subsampling: int = 64


# Optimizer information
@dataclass
class AdamWConfig(OptimizerInfo):
    """Optimizer configuration for BabyLM pretraining."""

    optimizer: Callable = AdamW
    optimizer_params: Mapping[str, Any] = field(
        default_factory=lambda: {
            "lr": 0,  # TODO: update for finetuning
            "weight_decay": 0.1,
            "betas": (0.9, 0.95),
            "fused": True,
        }
    )
    lr_scheduler: Callable = warmup_cosine_decay
    lr_scheduler_params: Mapping[str, Any] = field(
        default_factory=lambda: {
            "warmup_fraction": 0.07,
            "final_lr_ratio": 0.03,  # TODO: update for finetuning
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
    loglevel: int = logging.INFO
    seed: int = 42

    # Learning
    federated: int = 1
    federated_batches: int = 1
    federated_layer: Tuple[Type, ...] = (TransformerProcessorBlock,)
    epochs: int = 6

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
    def _pre_train_hook(
        model: nn.Module,
        state: DistributedState,
        config: XFFLConfig,
        epoch: int,
    ) -> Tuple[DataLoader, DataLoader, int]:
        ctx: AnemoiContext = ctx_dict[state.rank]  # type: ignore

        assert ctx.rollout_start is not None
        assert ctx.rollout_increment is not None
        assert ctx.rollout_max is not None

        if epoch > 0:
            assert ctx.current_rollout is not None

            ctx.current_rollout = min(
                ctx.current_rollout + ctx.rollout_increment, ctx.rollout_max
            )
        else:
            ctx.current_rollout = ctx.rollout_start

        splits, _ = get_dataset_splits(ctx=ctx, batch_size=config.dataset_info.batch_sizes, rollout=ctx.current_rollout)  # type: ignore
        dataloaders: Mapping[str, DataLoader] = create_dataloaders(
            state=state, dataset=splits, config=config
        )
        return dataloaders["train"], dataloaders["val"], ctx.current_rollout

    pre_train_hook: Callable = _pre_train_hook

    @staticmethod
    def _pre_process_hook(
        batch: Any,
        model: nn.Module,
        state: DistributedState,
        rstep: Optional[int] = None,
        rollout: Optional[int] = None,
        output: Optional[Tensor] = None,
        prev_data: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        return finetuning_pre_process_hook(model=model, batch=batch, state=state, rstep=rstep, rollout=rollout, output=output, prev_data=prev_data, ctx=ctx_dict[state.rank])  # type: ignore

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
            "mode": "disabled",  # "online" to active WandB
        }
    )

    # Output
    output_folder: Optional[Path] = None
    output_model: Optional[str] = None
