"""Common distributed PyTorch utilities"""

import os
import time
from logging import Logger, getLogger
from typing import Literal, Optional, Tuple

import torch.distributed as dist
from torch import cuda, nn
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.nn.functional import all_reduce

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


def federated_averaging(model: nn.Module, state: DistributedState) -> None:
    """Federated averaging of corresponding model's shards on different hosts

    :param model: PyTorch model
    :type model: nn.Module
    :param state: Partially instantiated distributed state (rank, world_size, backend)
    :type state: DistributedState
    """
    start_time = time.perf_counter()
    for param in model.parameters():  # TODO: raggruppare parametri?
        all_reduce(
            tensor=param,
            op=dist.ReduceOp.AVG,  # TODO: only on NCCL?
            group=state.federated_group,
        )
    logger.debug(f"Averaging time: {(time.perf_counter() - start_time):.2f} seconds")


def setup_distributed_process_group(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    group_local_rank: Optional[int] = None,
    group_local_size: Optional[int] = None,
    group_rank: Optional[int] = None,
    group_world_size: Optional[int] = None,
    backend: Optional[Literal["nccl", "gloo", "mpi"]] = "nccl",
    master_addr: Optional[str] = None,
    master_port: Optional[int] = None,
    federated_rank: Optional[int] = None,
    device: Optional[Literal["cpu", "cuda"]] = None,
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
        device=("cuda" if cuda.is_available() else "cpu") if device is None else device,
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
    if federated is None and "FEDERATED_LOCAL_WORLD_SIZE" in os.environ:
        federated = os.environ.get("FEDERATED_LOCAL_WORLD_SIZE")

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
            if federated_rank is None and "FEDERATED_RANK" in os.environ:
                federated_rank = int(os.environ.get("FEDERATED_RANK"))
            state.set_asymmetric_federated_scaling(
                federated_local_rank=state.rank % federated[federated_rank],
                federated_local_size=federated,
                federated_rank=federated_rank,
                federated_world_size=len(federated),
            )
    else:  # Setting non-federated techniques
        if hsdp is not None:
            # TODO: optimize this, create esh/groups only if necessary
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
