"""Common distributed PyTorch utilities"""

import os
from logging import Logger, getLogger
from typing import List, Literal, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import ShardingStrategy

from xffl.utils.distributed_state import DistributedState
from xffl.utils.utils import get_default_nccl_process_group_options, get_timeout

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def get_appropriate_sharding_strategy(state: DistributedState) -> ShardingStrategy:
    """Federated averaging of corresponding model's shards on different hosts

    :param state: Instantiated distributed state
    :type state: DistributedState
    """
    sharding_strategy: ShardingStrategy = (
        ShardingStrategy.HYBRID_SHARD
        if state.is_hsdp_setup()
        else ShardingStrategy.FULL_SHARD
    )
    logger.debug(
        f'[Rank {state.rank}]: Activating "{sharding_strategy}" sharding strategy'
    )

    return sharding_strategy


def sync_federated_averaging(model: nn.Module, state: DistributedState) -> None:
    """Federated averaging of corresponding model's shards on different hosts

    :param model: PyTorch model
    :type model: nn.Module
    :param state: Partially instantiated distributed state (rank, world_size, backend)
    :type state: DistributedState
    """
    replica_world_size: int = (
        min(state.replica_world_size)
        if not isinstance(state.replica_world_size, int)
        else state.replica_world_size
    )
    communicating_processes: int = (
        (state.replica_local_size // replica_world_size)
        if replica_world_size <= state.replica_local_size
        else 1
    )

    param_list = list(model.parameters())
    buffer = [
        param_list[0], # Tensor 0 has different dimensions
        torch.stack(param_list[1:]),
    ]

    if (
        communicating_processes * state.replica_rank
        <= state.replica_local_rank
        < communicating_processes * (state.replica_rank + 1)
    ):
        for param in buffer:
            dist.all_reduce(
                tensor=param.contiguous(),
                op=dist.ReduceOp.AVG,
                group=state.federated_group,
            )
            dist.broadcast(
                tensor=param.contiguous(),
                src=state.rank,
                group=state.replica_group,
            )
    else:
        federated_local_size: int = (
            state.federated_local_size[state.federated_rank]
            if not isinstance(state.federated_local_size, int)
            else state.federated_local_size
        )
        src: int = (
            state.replica_local_rank + (federated_local_size * state.federated_rank)
            if state.replica_local_rank < communicating_processes * state.replica_rank
            else state.replica_local_rank
            + (state.replica_local_size * (state.replica_rank + 1))
            + (federated_local_size * state.federated_rank)
        )

        for param in buffer:
            dist.broadcast(
                tensor=param.contiguous(),
                src=src,
                group=state.replica_group,
            )

    for param, updated_param in zip(param_list[1:], buffer[1]):
        param.copy_(updated_param, non_blocking=True)


def async_federated_averaging(
    model: nn.Module, state: DistributedState
) -> Tuple[List[dist.Work], List[torch.Tensor]]:
    """Federated averaging of corresponding model's shards on different hosts

    :param model: PyTorch model
    :type model: nn.Module
    :param state: Partially instantiated distributed state (rank, world_size, backend)
    :type state: DistributedState
    """
    replica_world_size: int = (
        min(state.replica_world_size)
        if not isinstance(state.replica_world_size, int)
        else state.replica_world_size
    )
    communicating_processes: int = (
        (state.replica_local_size // replica_world_size)
        if replica_world_size <= state.replica_local_size
        else 1
    )

    param_list = list(model.parameters())
    buffer = [
        param_list[0],  # Tensor 0 has different dimensions
        torch.stack(param_list[1:]),
    ]

    # pre_values = []
    # post_values = []
    # differences = []

    # for p in model.parameters():
    #    if p.requires_grad:  # only change learnable parameters
    #        with torch.no_grad():
    #            p.fill_(state.federated_rank)

    # for param in model.parameters():
    #    pre_values.append(param.clone().detach().cpu())

    all_reduce_handles: List[dist.Work] = []
    broadcast_handles: List[dist.Work] = []
    if (
        communicating_processes * state.replica_rank
        <= state.replica_local_rank
        < communicating_processes * (state.replica_rank + 1)
    ):
        for param in buffer:
            all_reduce_handles.append(
                dist.all_reduce(
                    tensor=param.contiguous(),
                    op=dist.ReduceOp.AVG,
                    group=state.federated_group,
                    async_op=True,
                )
            )
        for all_reduce_handle, param in zip(
            all_reduce_handles, buffer
        ):
            all_reduce_handle.wait(timeout=get_timeout(seconds=60))
            broadcast_handles.append(
                dist.broadcast(
                    tensor=param.contiguous(),
                    src=state.rank,
                    group=state.replica_group,
                    async_op=True,
                )
            )
    else:
        federated_local_size: int = (
            state.federated_local_size[state.federated_rank]
            if not isinstance(state.federated_local_size, int)
            else state.federated_local_size
        )
        src: int = (
            state.replica_local_rank + (federated_local_size * state.federated_rank)
            if state.replica_local_rank < communicating_processes * state.replica_rank
            else state.replica_local_rank
                 + (state.replica_local_size * (state.replica_rank + 1))
                 + (federated_local_size * state.federated_rank)
        )

        for param in buffer:
            broadcast_handles.append(
                dist.broadcast(
                    tensor=param.contiguous(),
                    src=src,
                    group=state.replica_group,
                    async_op=True,
                )
            )

    return broadcast_handles, buffer

    # for param in model.parameters():
    #    post_values.append(param.clone().detach().cpu())

    # for layer, _ in enumerate(model.parameters()):
    #    differences.append(pre_values[layer] - post_values[layer])

    # logger.warning(f"[RANK {state.rank}]:\n"
    #               f"pre-values: {pre_values}\n"
    #               f"post-values: {post_values}\n"
    #               f"differences: {differences}\n")


def setup_distributed_process_group(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    group_local_rank: Optional[int] = None,
    group_local_size: Optional[int] = None,
    group_rank: Optional[int] = None,
    group_world_size: Optional[int] = None,
    backend: Literal["nccl", "gloo", "mpi"] = "nccl",
    master_addr: Optional[str] = None,
    master_port: Optional[int] = None,
    federated_rank: Optional[int] = None,
    device: Literal["cpu", "cuda"] = "cuda",
    hsdp: Optional[int] = None,
    federated: Optional[int | Tuple[int]] = None,
) -> DistributedState:
    """Setup PyTorch's distributed environment

    To be called AFTER the various processes have been created and by ALL processes
    The distributed rendez-vous point determined by two environmental variable that should be set BEFORE calling this method (same values for all processes):
    MASTER_ADD:  network address of the rendezvous
    MASTER_PORT: network port of the rendezvous

    :param rank: Rank of the calling process, otherwise obtained from the environment, defaults to None
    :type rank: Optional[int], optional
    :param world_size: Global world size, otherwise obtained from the environment, defaults to None
    :type world_size: Optional[int], optional
    :param group_local_rank: Local group rank of the calling process, otherwise obtained from the environment, defaults to None
    :type group_local_rank: Optional[int], optional
    :param group_local_size: Size of the local group of the calling process, otherwise obtained from the environment, defaults to None
    :type group_local_size: Optional[int], optional
    :param group_rank: Rank of the group of the calling process, otherwise obtained from the environment, defaults to None
    :type group_rank: Optional[int], optional
    :param group_world_size: World size of the groups, otherwise obtained from the environment, defaults to None
    :type group_world_size: Optional[int], optional
    :param backend: Communication backend to be used, defaults to "nccl" (distributed GPU training)
    :type backend: Optional[Literal["nccl", "gloo", "mpi"]], optional
    :param master_addr: IP address for the PyTorch.distributed rendez-vous, otherwise obtained from the environment, defaults to None
    :type master_addr: Optional[str], optional
    :param federated_rank: ID of the current network cell, defaults to None
    :type federated_rank: Optional[int], optional
    :param master_port: Port number for the PyTorch.distributed rendez-vous, otherwise obtained from the environment, defaults to None
    :type master_port: Optional[int], optional
    :param device: Device type used by the distributed processes, if not specified we will try to guess it, defaults to None
    :type device: Optional[Literal["cpu", "cuda"]], optional
    :param hsdp: Activate Hybrid Sharding Distributed Parallelism with specified replica group size, defaults to None
    :type hsdp: Optional[int], optional
    :param federated:Activate Federated Scaling with specified federated group size, defaults to None
    :type federated: Optional[int], optional
    :raises AttributeError: If backend is not in nccl, gloo, or mpi
    :raises ValueError: If no valid MASTER_ADDR and MASTER_PORT are set
    :return: Distributed state of the current training setup
    :rtype: DistributedState
    """
    state: DistributedState = DistributedState()

    # Distributed state technical information
    state.set_technical(
        backend=backend,
        master_addr=(
            os.environ.get("MASTER_ADDR") if master_addr is None else master_addr
        ),
        master_port=(
            int(os.environ.get("MASTER_PORT")) if master_port is None else master_port
        ),
        device=device,
    )

    # Distributed state global information
    state.set_global(
        rank=int(os.environ.get("RANK")) if rank is None else rank,
        world_size=int(
            os.environ.get("WORLD_SIZE") if world_size is None else world_size
        ),
    )

    # Basic PyTorch distributed setup
    init_distributed_process_group(state=state)

    # Distributed state local information
    state.set_group(
        group_local_rank=(
            int(os.environ.get("LOCAL_RANK"))
            if group_local_rank is None
            else group_local_rank
        ),
        group_local_size=(
            int(os.environ.get("LOCAL_WORLD_SIZE"))
            if group_local_size is None
            else group_local_size
        ),
        group_rank=(
            int(os.environ.get("GROUP_RANK")) if group_rank is None else group_rank
        ),
        group_world_size=(
            int(os.environ.get("GROUP_WORLD_SIZE"))
            if group_world_size is None
            else group_world_size
        ),
    )

    # Setting HSDP if needed
    if hsdp is not None:
        state.set_hsdp(
            replica_local_rank=state.rank % hsdp,
            replica_local_size=hsdp,
            replica_rank=state.rank // hsdp,
            replica_world_size=state.world_size // hsdp,
        )

    # Check if asymmetric Federated Scaling is required #
    if federated is None and "XFFL_FEDERATED_LOCAL_WORLD_SIZE" in os.environ:
        federated = tuple(
            int(item) * state.group_local_size
            for item in os.environ.get("XFFL_FEDERATED_LOCAL_WORLD_SIZE").split(",")
        )

    if federated is not None:
        symmetric_federated_scaling: bool = isinstance(federated, int)
        asymmetric_federated_scaling: bool = isinstance(federated, Tuple)

        if symmetric_federated_scaling and asymmetric_federated_scaling:
            logger.warning(
                "Both symmetric and asymmetric federated scaling are enabled - falling back to symmetric federated scaling"
            )
            asymmetric_federated_scaling = False

        if symmetric_federated_scaling:  # Symmetric federation
            state.set_symmetric_federated_scaling(
                federated_local_rank=state.rank % federated,
                federated_local_size=federated,
                federated_rank=state.rank // federated,
                federated_world_size=state.world_size // federated,
            )
        elif asymmetric_federated_scaling:  # Asymmetric federation
            if federated_rank is None and "XFFL_FEDERATED_RANK" in os.environ:
                federated_rank = int(os.environ.get("XFFL_FEDERATED_RANK"))
            state.set_asymmetric_federated_scaling(
                federated_local_rank=state.rank % federated[federated_rank],
                federated_local_size=federated,
                federated_rank=federated_rank,
                federated_world_size=len(federated),
            )
    else:  # Setting non-federated techniques
        if hsdp is not None:
            state.set_hsdp_mesh()
        else:
            state.set_fsdp_mesh()

    logger.debug(f"[Rank {state.rank}]: distributed setup: {state}")
    return state


def init_distributed_process_group(state: DistributedState) -> None:
    """PyTorch's distributed backend initialization

    :param state: Partially instantiated distributed state (rank, world_size, backend)
    :type state: DistributedState
    """
    dist.init_process_group(
        backend=state.backend,
        world_size=state.world_size,
        rank=state.rank,
        pg_options=(
            get_default_nccl_process_group_options()
            if state.backend == "nccl"
            else None
        ),
        timeout=get_timeout(),
    )


def setup_hsdp_mesh(state: DistributedState, hsdp: int) -> None:

    state.set_hsdp(
        replica_local_rank=state.rank % hsdp,
        replica_local_size=hsdp,
        replica_rank=state.rank // hsdp,
        replica_world_size=state.world_size // hsdp,
    )


def cleanup_distributed_process_group(state: DistributedState) -> None:
    """Cleanup PyTorch's distributed environment

    To be called AFTER the various processes have completed their work and by ALL processes

    :param state: Instantiated distributed state
    :type state: DistributedState
    """
    logger.debug(f"[Rank {state.rank}]: calling destroy_process_group")

    dist.destroy_process_group(dist.GroupMember.WORLD)
