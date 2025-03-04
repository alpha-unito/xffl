"""Methods useful for model handling"""

import os
from logging import Logger, getLogger
from pathlib import Path
from typing import Optional

import torch
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.optim import Optimizer

from xffl.custom.types import PathLike

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def save_FSDP_model(
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

    # Gather the full, unsharded state dict
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
            safe_serialization=True,  # Safeternsor or Pickle
        )  # Shard size can be controlled (can improve transfer performance)

        logger.info(f"Model saved to {save_path}")
