"""Methods useful for model handling"""

import os
from logging import Logger, getLogger
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
from torch.optim import Optimizer
from transformers import AutoModel

from xffl.custom.models import ModelInfo
from xffl.custom.types import PathLike
from xffl.distributed.distributed import (
    DistributedState,
    get_appropriate_sharding_strategy,
)

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def create_fsdp_model(
    module: nn.Module | AutoModel,
    state: DistributedState,
    model_info: ModelInfo,
    mixed_precision: Optional[MixedPrecision] = None,
) -> FullyShardedDataParallel:
    """Creates an FSDP model

    :param module: FSDP-wrapped model to be saved
    :type module: nn.Module | AutoModel
    :param state: Instantiated distributed state
    :type state: DistributedState
    :param model_info: Dataclass with model's information
    :type model_info: ModelInfo
    :param mixed_precision: Precision to use for the module, defaults to None
    :type mixed_precision: Optional[MixedPrecision], optional
    :return: The original module wrapped by FSDP
    :rtype: FullyShardedDataParallel
    """

    device_mesh: Optional[bool] = None
    if state.is_hsdp_setup():  # TODO: Add 2D FSDP-TP parallelism support
        device_mesh = state.hsdp_mesh
    elif state.is_fsdp_setup():
        device_mesh = state.fsdp_mesh

    model: FullyShardedDataParallel = FullyShardedDataParallel(
        module=module,
        sharding_strategy=get_appropriate_sharding_strategy(state=state),
        auto_wrap_policy=model_info.wrapping_policy,
        device_id=state.current_device,
        forward_prefetch=True,
        limit_all_gathers=False,
        mixed_precision=mixed_precision,
        sync_module_states=state.meta_initialization,
        param_init_fn=lambda layer: (
            layer.to_empty(device=state.current_device, recurse=False)
            if state.meta_initialization
            else None
        ),
        device_mesh=device_mesh,
    )

    return model


def save_fsdp_model(
    model: FullyShardedDataParallel,  # To be generalized (as for now just HF)
    optimizer: Optimizer,
    path: PathLike,
    name: str,
    rank: int,
    epoch: Optional[int] = None,
    checkpoint: Optional[int] = None,
    precision: Optional[torch.dtype] = None,
) -> None:
    """Saves an FSDP wrapped model to a specified path

    All processes part of the FSDP training have to call this method to effectively save the model

    :param model: FSDP-wrapped model to be saved
    :type model: FullyShardedDataParallel
    :param optimizer: Model optimizer
    :type optimizer: Optimizer
    :param path: Path where to save the model
    :type path: PathLike
    :param name: Model's name
    :type name: str
    :param rank: Calling process' global rank
    :type rank: int
    :param epoch: Epoch tag to add to the model's name, defaults to None
    :type epoch: Optional[int], optional
    :param checkpoint: Checkpoint tag to add to the model's name, defaults to None
    :type checkpoint: Optional[int], optional
    :param precision: Numerical format to use to save the model, defaults to None
    :type precision: Optional[torch.dtype], optional
    """

    # Gather the full, un-sharded state dict
    state_dict, _ = get_state_dict(
        model=model,
        optimizers=optimizer,
        options=StateDictOptions(
            full_state_dict=True,
        ),
    )

    # Only rank 0 saves the model
    if rank == 0:
        # Saving path creation
        if not os.path.exists(path):
            raise Exception(f"Save model path {path} does not exist")
        if not os.path.isdir(path):
            raise Exception(f"Save model path {path} must be a directory")
        if epoch:
            save_path = Path(path, name, f"epoch_{epoch}")
        elif checkpoint:
            save_path = Path(path, name, f"checkpoint_{checkpoint}")
        else:
            save_path = Path(path, name)
        save_path.mkdir(parents=True, exist_ok=True)

        # Saving state_dict changing precision
        if precision:
            for param_tensor in state_dict:
                state_dict[param_tensor] = state_dict[param_tensor].to(
                    device="cpu",
                    dtype=precision,
                    non_blocking=True,
                    # Possible to specify the memory format
                )

        # This is HF specific (modelling_utils.py)
        # Saves the model (torch.save) together with its configuration files
        # so that it can be reloaded with PreTrainedModel.from_pretrained
        model.save_pretrained(
            save_directory=save_path,
            state_dict=state_dict,
            safe_serialization=True,  # Safetensor or Pickle
        )  # Shard size can be controlled (can improve transfer performance)

        logger.info(f"Model saved to {save_path}")
