"""Methods useful for model handling
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel,
    StateDictType,
)
from transformers import PreTrainedModel

from xffl.custom.types import PathLike


def save_FSDP_model(
    model: FullyShardedDataParallel, # To be generalized (as for now just HF)
    path: PathLike,
    name: str,
    rank: int,
    epoch: Optional[int] = None,
    checkpoint: Optional[int] = None,
    precision: Optional[torch.dtype] = None,
    verbose: Optional[bool] = None,
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
    :param verbose: Enable verbose output, defaults to None
    :type verbose: Optional[bool], optional
    """

    # Ensure that training is done on all ranks
    dist.barrier()

    # Gather the full, unsharded state dict on rank 0 process
    with FullyShardedDataParallel.state_dict_type(
        module=model,
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        # optim_state_dict_config=OptimStateDictConfig(offload_to_cpu=True),  # TODO: Save also optimizer?
    ):
        state_dict: Dict[str, torch.Tensor] = model.state_dict()

    # Only rank 0 saves the model
    if rank == 0:
        # Saving path creation
        if epoch:
            save_path = f"{path}/{name}/epoch_{epoch}/"
        elif checkpoint:
            save_path = f"{path}/{name}/checkpoint_{checkpoint}/"
        else:
            save_path = f"{path}/{name}/"

        save_path = Path(save_path)
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
        # Saves the model (torch.save) togheter with its configuration files
        # so that it can be reloaded with PreTrainedModel.from_pretrained
        model.save_pretrained(
            save_directory=save_path,
            state_dict=state_dict,
            safe_serialization=True,  # Safeternsor or Pickle
        )  # Shard size can be controlled (can improve transfer performance)

        if verbose:
            print(f"Model saved to {save_path}")
