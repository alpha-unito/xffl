"""Common distributed PyTorch utilities"""

import math
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from logging import Logger, getLogger
from typing import Literal, Optional, Tuple

import torch
import torch.distributed as dist
from torch import cuda, nn
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.nn.functional import all_reduce

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


@dataclass
class DistributedState:
    """This dataclass traces all the distributed environment parameters"""

    # GLOBAL
    rank: Optional[int] = None
    """Global rank"""
    world_size: Optional[int] = None
    """Global world size"""

    # LOCAL GROUP
    group_local_rank: Optional[int] = None
    """Local rank"""
    group_local_size: Optional[int] = None
    """Local group size"""
    group_rank: Optional[int] = None
    """Group rank"""
    group_world_size: Optional[int] = None
    """Group world size"""

    # REPLICA GROUP
    replica_local_rank: Optional[int] = None
    """Local rank"""
    replica_local_size: Optional[int] = None
    """Local replica size"""
    replica_rank: Optional[int] = None
    """Replica rank"""
    replica_world_size: Optional[int] = None
    """Replica world size"""

    # FEDERATED GROUP
    federated_local_rank: Optional[int] = None
    """Local rank"""
    federated_local_size: Optional[int] = None
    """Local group size"""
    federated_rank: Optional[int] = None
    """Group rank"""
    federated_world_size: Optional[int] = None
    """Group world size"""

    # MESH
    fsdp_mesh: Optional[DeviceMesh] = None
    """FSDP device mesh"""
    hsdp_mesh: Optional[DeviceMesh] = None
    """HSDP device mesh"""
    federated_group: Optional[ProcessGroup] = None
    """Federated scaling device mesh"""

    # TECHNICAL
    backend: Optional[Literal["nccl", "gloo", "mpi"]] = None
    """Communication backend"""
    master_addr: Optional[str] = None
    """Rendez-vous address"""
    master_port: Optional[int] = None
    """Rendez-vous port"""
    device: Optional[Literal["cpu", "gpu"]] = None
    """Chosen deployment device"""


def get_appropriate_sharding_strategy(state: DistributedState) -> ShardingStrategy:
    """Federated averaging of corresponding model's shards on different hosts

    :param state: Instantiated distributed state
    :type state: DistributedState
    """
    sharding_strategy: ShardingStrategy = (
        ShardingStrategy.HYBRID_SHARD
        if state.hsdp_mesh
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


def get_timeout(seconds: Optional[int] = 60) -> timedelta:
    """Maximum allowed timeout for distributed communnications

    :param seconds: Maximum allowed timeout in seconds, defaults to 60
    :type seconds: Optional[int], optional
    :return: Maximum allowed time delta
    :rtype: timedelta
    """
    return timedelta(seconds=seconds)


def get_deafult_nccl_process_group_options(
    is_high_priority_stream: Optional[bool] = True,
):
    """Default NCCL backend configuration for xFFL

    :param is_high_priority_stream: Wether to pick up the highest priority CUDA stream, defaults to True
    :type is_high_priority_stream: Optional[bool], optional
    :return: Configured options for the NCCL backend
    :rtype: ProcessGroupNCCL.Options
    """
    from torch.distributed import ProcessGroupNCCL

    options: ProcessGroupNCCL.Options = ProcessGroupNCCL.Options()
    options.is_high_priority_stream = is_high_priority_stream
    options._timeout = get_timeout()

    return options


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
    device: Optional[Literal["cpu", "gpu"]] = None,
    hsdp: Optional[int] = None,
    federated: Optional[int] = None,
) -> Tuple[int, int, int, Optional[DeviceMesh]]:
    """Setup PyTorch's distributed environment

    To be called AFTER the various processes have been created and by ALL processes
    The distributed randevouz point is determinated by two environmental variable that should be set BEFORE calling this method (same values for all processes):
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
    :param master_port: Port number for the PyTorch.distributed rendez-vous, otherwise obtained from the environment, defaults to None
    :type master_addr: Optional[int], optional
    :param device: Device type used by the distributed processes, if not specified we will try to guess it, defaults to None
    :type device: Optional[Literal["cpu", "gpu"]], optional
    :param hsdp: Activate Hybrid Sharding Distributed Parallelism with specified replica group size, defaults to 0
    :type hsdp: Optional[int], optional
    :param federated: Wether to activate Federated Scaling, defaults to False
    :type federated: Optional[bool], optional
    :raises AttributeError: If backend is not in nccl, gloo, or mpi
    :raises ValueError: If no valid MASTER_ADDR and MASTER_PORT are set
    :return: Rank and local rank of the calling process, global world size, HSDP device mesh, and federated scaling groups
    :rtype: Tuple[int, int, int, Optional[DeviceMesh], Optional[List[ProcessGroup]]]
    """
    state: DistributedState = DistributedState()

    # Distributed state global information
    state.rank = rank if rank else int(os.environ.get("RANK"))
    state.world_size = world_size if world_size else int(os.environ.get("WORLD_SIZE"))
    logger.debug(f"[Rank {state.rank}]: Setting up distributed environment")

    # Distributed state technical information
    state.backend = backend
    state.master_addr = master_addr if master_addr else os.environ.get("MASTER_ADDR")
    state.master_port = (
        master_port if master_port else int(os.environ.get("MASTER_PORT"))
    )
    state.device = device if device else ("cuda" if cuda.is_available() else "cpu")

    if not (state.master_addr or state.master_port):
        raise ValueError(
            f"No valid master address ({state.master_addr}) and/or master port ({master_port}) specified, setting up distributed environment impossible."
        )

    # Distributed state local information
    state.group_local_rank = (
        group_local_rank if group_local_rank else int(os.environ.get("LOCAL_RANK"))
    )
    state.group_local_size = (
        group_local_size
        if group_local_size
        else int(os.environ.get("LOCAL_WORLD_SIZE"))
    )
    state.group_rank = group_rank if group_rank else int(os.environ.get("GROUP_RANK"))
    state.group_world_size = (
        group_world_size
        if group_world_size
        else int(os.environ.get("GROUP_WORLD_SIZE"))
    )

    # Basic PyTorch distributed setup
    init_distributed_process_group(state=state)

    if hsdp:
        state.replica_local_size = hsdp

    if federated:
        # Federated Scaling process groups - Sharding intra-group, replicating inter-groups
        state.federated_local_size = federated
        setup_federated_scaling_groups(state=state)
    elif hsdp:
        # 2D device mesh creation for HSDP - Sharding intra-group, replicating inter-groups
        setup_hsdp_mesh(state=state)

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
            get_deafult_nccl_process_group_options()
            if state.backend == "nccl"
            else None
        ),
        timeout=get_timeout(),
    )


def setup_hsdp_mesh(state: DistributedState) -> None:
    """Creates a 2D device mesh allowing Hybrid Sharding Data Parallel (HSDP) training

    :param state: Instantiated distributed state
    :type state: DistributedState
    """
    replica_world_size: int = state.world_size // state.replica_local_size

    if replica_world_size > 1 and state.world_size % state.replica_local_size == 0:
        state.hsdp_mesh = init_device_mesh(
            device_type=state.device,
            mesh_shape=[replica_world_size, state.replica_local_size],
            mesh_dim_names=["replica", "shard"],
        )

        state.replica_world_size = replica_world_size
        state.replica_local_rank = state.rank % state.replica_local_size
        state.replica_rank = state.rank // state.replica_local_size
    else:
        logger.error(
            f"Impossible setting up HSDP device mesh with replica over {state.replica_local_size} processes and {state.replica_world_size} replicas: world size is not divisible by replica local size into multiple replica groups.\nFalling back to standard FSDP."
        )
        state.replica_local_size = None


def setup_federated_scaling_groups(state: DistributedState) -> None:
    """Create the federated scaling rank groups

    This process groups bring together all the ranks handling corresponding model's shards.
    E.g.: if a model is sharded among four processes and replicated across two process groups (i.e., device_mesh=[[0,1,2,3],[4,5,6,7]])
    then the federated scaling process groups correspond to the groups of processes having the same local rank (i.e., [[0,4][1,5][2,6][3,7]])

    :param state: Instantiated distributed state
    :type state: DistributedState
    """
    federated_world_size = state.world_size // state.federated_local_size

    if federated_world_size > 1 and state.world_size % state.federated_local_size == 0:
        state.federated_world_size = federated_world_size
        state.federated_local_rank = state.rank % state.federated_local_size
        state.federated_rank = state.rank // state.federated_local_size

        if state.replica_local_size:
            replica_world_size: int = (
                state.federated_local_size // state.replica_local_size
            )

            if (
                replica_world_size > 1
                and state.federated_world_size % state.replica_local_size == 0
            ):
                state.replica_world_size = replica_world_size
                state.replica_local_rank = state.rank % state.replica_local_size
                state.replica_rank = (
                    state.federated_local_rank // state.replica_local_size
                )

                mesh: torch.Tensor = create_device_mesh(
                    (
                        state.federated_world_size,
                        state.replica_world_size,
                        state.replica_local_size,
                    )
                )

                state.hsdp_mesh = DeviceMesh(
                    device_type=state.device,
                    mesh=mesh[state.federated_rank],
                    mesh_dim_names=["replica", "shard"],
                    _init_backend=False,
                )

                state.federated_group = create_process_group(
                    ranks=mesh[:, state.replica_rank, state.replica_local_rank],
                    state=state,
                    group_desc=f"Federated shard averaging group #{state.federated_rank}",
                )
            else:
                mesh: torch.Tensor = create_device_mesh(
                    (
                        state.federated_world_size,
                        state.world_size // state.federated_world_size,
                    )
                )

                state.fsdp_mesh = DeviceMesh.from_group(
                    device_type=state.device,
                    group=create_process_group(
                        ranks=mesh[state.federated_rank],
                        state=state,
                        group_desc=f"Federated sharding group #{state.federated_rank}",
                    ),
                    mesh_dim_names=["shard"],
                )

                state.federated_group = create_process_group(
                    ranks=mesh[:, state.federated_local_rank],
                    state=state,
                    group_desc=f"Federated shard averaging group #{state.federated_rank}",
                )
    else:
        logger.error(
            f"Impossible setting up Federated Scaling device mesh with {state.federated_local_size} processes per federated group: world size is not divisible by federated local size into into multiple federated groups.\nFalling back to standard HSDP."
        )
        state.federated_local_size = None


def create_process_group(
    ranks: Tuple[int, ...], group_desc: Optional[str], state: DistributedState
) -> ProcessGroup:
    """Creates a new process group with the specified ranks

    Only the interested rank can enter this method

    :param ranks: Ranks making up the group
    :type ranks: Tuple[int, ...]
    :param group_desc: Description of the process group
    :type group_desc: Optional[str]
    :param state: Instantiated distributed state
    :type state: DistributedState
    :return: Process group handle
    :rtype: ProcessGroup
    """
    return dist.new_group(
        ranks=ranks,
        timeout=get_timeout(),
        backend=state.backend,
        pg_options=get_deafult_nccl_process_group_options(),
        use_local_synchronization=True,
        group_desc=group_desc,
    )


def create_device_mesh(mesh_shape: Tuple[int, ...]) -> torch.Tensor:
    """Creates a Tensor of distributed process ranks with the specified dimensions

    :param mesh_shape: Dimensions of the mesh
    :type mesh_shape: Tuple[int, ...]
    :return: Tensor of ranks
    :rtype: torch.Tensor
    """
    return torch.arange(math.prod(mesh_shape), dtype=torch.int).view(mesh_shape)


def cleanup_distributed_process_group(state: DistributedState) -> None:
    """Cleanup PyTorch's distributed environment

    To be called AFTER the various processes have completed their work and by ALL processes

    :param state: Instantiated distributed state
    :type state: DistributedState
    """
    logger.debug(f"[Rank {state.rank}]: calling destroy_process_group")

    dist.destroy_process_group(dist.GroupMember.WORLD)
