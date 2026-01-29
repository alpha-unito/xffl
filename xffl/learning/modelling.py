"""Methods useful for model handling"""

import os
from logging import Logger, getLogger
from pathlib import Path
from typing import Callable, Optional, Type

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
from torch.optim import Optimizer

from xffl.custom.config import XFFLConfig
from xffl.distributed.distributed import (
    DistributedState,
    get_appropriate_sharding_strategy,
)
from xffl.learning import utils
from xffl.utils.utils import resolve_param

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def create_fsdp_model(
    state: DistributedState,
    module: Optional[nn.Module] = None,
    wrapping_policy: Optional[Callable] = None,
    mixed_precision: Optional[MixedPrecision] = None,
    decoder_layers: Optional[Type] = None,
    config: Optional[XFFLConfig] = None,
    use_orig_params: bool = False,
) -> FullyShardedDataParallel | nn.Module:  # TODO: Move to FSDP2
    """Creates an FSDP model

    The parameters can be provided both directly and through an XFFL configuration.
    In case both are provided, the firsts take the precedence.

    :param module: FSDP-wrapped model to be saved
    :type module: nn.Module
    :param state: xFFL distributed state
    :type state: DistributedState
    :param wrapping_policy: Model's wrapping policy, defaults to None
    :type wrapping_policy:  Optional[Callable], optional
    :param mixed_precision: Precision to use for the module, defaults to None
    :type mixed_precision: Optional[MixedPrecision], optional
    :param decoder_layers: Layer type to set activation checkpointing, defaults to None
    :type decoder_layers: Optional[Type], optional
    :param use_orig_params: If to use the original parameter format, defaults to False
    :type use_orig_params: Bool, defaults to False
    :param config: XFFL configuration
    :type config: Optional[XFFLConfig], defaults to None
    :return: The original module wrapped by FSDP
    :rtype: FullyShardedDataParallel
    """

    # Parameters resolution
    if config is not None:
        if module is None:
            __module: Optional[Callable] = resolve_param(
                value=module, config=config.model_info, attr="model"
            )
            if __module is not None:
                _module: Optional[nn.Module] = __module(config=config, state=state)
        _wrapping_policy: Optional[Callable] = resolve_param(
            value=wrapping_policy, config=config.model_info, attr="wrapping_policy"
        )
        _mixed_precision: Optional[MixedPrecision] = resolve_param(
            value=mixed_precision, config=config.model_info, attr="mixed_precision"
        )
        _decoder_layers: Optional[Type] = resolve_param(
            value=decoder_layers, config=config.model_info, attr="decoder_layers"
        )
    else:
        _module: Optional[nn.Module] = module
        _wrapping_policy: Optional[Callable] = wrapping_policy
        _mixed_precision: Optional[MixedPrecision] = mixed_precision
        _decoder_layers: Optional[Type] = decoder_layers

    # Model and device mashes creation
    if _module is not None:

        model: FullyShardedDataParallel | nn.Module

        if dist.is_initialized():
            device_mesh: Optional[DeviceMesh] = None
            if state.is_hsdp_setup():  # TODO: Add 2D FSDP-TP parallelism support
                device_mesh = state.hsdp_mesh
            elif state.is_fsdp_setup():
                device_mesh = state.fsdp_mesh

            model = FullyShardedDataParallel(
                module=_module,
                sharding_strategy=get_appropriate_sharding_strategy(state=state),
                auto_wrap_policy=_wrapping_policy,
                device_id=state.current_device,
                forward_prefetch=True,
                limit_all_gathers=False,
                mixed_precision=_mixed_precision,
                sync_module_states=bool(state.meta_initialization),
                param_init_fn=lambda layer: (
                    layer.to_empty(device=state.current_device, recurse=False)
                    if state.meta_initialization
                    else None
                ),  # type: ignore
                device_mesh=device_mesh,
                use_orig_params=use_orig_params,
            )
        else:
            model = _module.to(device=state.current_device, non_blocking=True)

        # Activation checkpointing
        # This can also be called before FSDP, will result in applying the HF-specific method, giving warnings during the training
        if _decoder_layers is not None:  # TODO: make this optional/paramterizable
            utils.set_activation_checkpointing(
                model=model,
                layer=_decoder_layers,
            )

    else:
        logger.critical(
            "Impossible setting up the distributed training: no model provided."  # TODO: add an exception?
        )

    return model


def save_model(
    model: (
        nn.Module | FullyShardedDataParallel
    ),  # To be generalized (as for now just HF)
    optimizer: Optimizer,
    path: Path,
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
    :type path: Path
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

    # Clear GPU cache and reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Gather the full, un-sharded state dict
    state_dict, _ = get_state_dict(
        model=model,
        optimizers=optimizer,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
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
                state_dict[param_tensor] = state_dict[param_tensor].to(  # type: ignore
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
        )  # type: ignore # Shard size can be controlled (can improve transfer performance)

        logger.info(f"Model saved to {save_path}")
