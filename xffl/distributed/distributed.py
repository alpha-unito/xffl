"""Common distributed PyTorch utilities"""

import os
from logging import Logger, getLogger
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import Backend
from torch.distributed.fsdp import ShardingStrategy

from xffl.distributed.distributed_state import DistributedState
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


def setup_distributed_process_group(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    group_local_rank: Optional[int] = None,
    group_local_size: Optional[int] = None,
    group_rank: Optional[int] = None,
    group_world_size: Optional[int] = None,
    backend: Optional[Backend] = None,
    master_addr: Optional[str] = None,
    master_port: Optional[int] = None,
    device: Optional[torch.device] = None,
    hsdp: Optional[int] = None,
    federated: Optional[Tuple[int]] = None,
    streams: Optional[int] = None,
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
    :param backend: Communication backend to be used, defaults to "nccl" (distributed GPU training), defaults to None
    :type backend: Backend, optional
    :param master_addr: IP address for the PyTorch.distributed rendez-vous, otherwise obtained from the environment, defaults to None
    :type master_addr: Optional[str], optional
    :param master_port: Port number for the PyTorch.distributed rendez-vous, otherwise obtained from the environment, defaults to None
    :type master_port: Optional[int], optional
    :param device: Device type used by the distributed processes, if not specified we will try to guess it, defaults to None
    :type device: Optional[torch.device], optional
    :param hsdp: Activate Hybrid Sharding Distributed Parallelism with specified replica group size, defaults to None
    :type hsdp: Optional[int], optional
    :param federated:Activate Federated Scaling with specified federated group size, defaults to None
    :type federated: Optional[int], optional
    :param streams: Number of CUDA streams to instantiate, defaults to None
    :type streams: Optional[int]
    :raises AttributeError: If backend is not in nccl, gloo, or mpi
    :raises ValueError: If no valid MASTER_ADDR and MASTER_PORT are set
    :return: Distributed state of the current training setup
    :rtype: DistributedState
    """
    state: DistributedState = DistributedState()

    # Distributed state global information
    state.set_global(
        backend=(
            backend
            if backend is not None
            else Backend("nccl" if torch.cuda.is_available() else "gloo")
        ),
        device_type=(
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ),
        master_addr=(
            os.environ.get("MASTER_ADDR") if master_addr is None else master_addr
        ),
        master_port=(
            int(os.environ.get("MASTER_PORT")) if master_port is None else master_port
        ),
        rank=int(os.environ.get("RANK")) if rank is None else rank,
        world_size=int(
            os.environ.get("WORLD_SIZE") if world_size is None else world_size
        ),
    )

    # Distributed state local information
    state.set_node(
        node_local_rank=(
            int(os.environ.get("LOCAL_RANK"))
            if group_local_rank is None
            else group_local_rank
        ),
        node_local_size=(
            int(os.environ.get("LOCAL_WORLD_SIZE"))
            if group_local_size is None
            else group_local_size
        ),
        node_rank=(
            int(os.environ.get("GROUP_RANK")) if group_rank is None else group_rank
        ),
        node_world_size=(
            int(os.environ.get("GROUP_WORLD_SIZE"))
            if group_world_size is None
            else group_world_size
        ),
    )

    # Setting execution device
    state.set_exec_device(current_device=_get_current_device(state=state))

    # Basic PyTorch distributed setup
    init_distributed_process_group(state=state)

    # Check if Federated Scaling is required
    if federated is None:
        if "XFFL_FEDERATED_LOCAL_WORLD_SIZE" in os.environ:
            federated = tuple(
                int(item) * state.node_local_size
                for item in os.environ.get("XFFL_FEDERATED_LOCAL_WORLD_SIZE").split(",")
            )
    elif len(federated) == 1:
        if state.world_size % federated[0] != 0:
            logger.error(
                f"The world size {state.world_size} is not divisible by the specified federated group size {federated[0]} - falling back to standard FSDP/HSDP"
            )
            federated = None
        elif state.world_size == federated[0]:
            logger.error(
                f"The world size {state.world_size} and the the specified federated group size {federated[0]} are equal - falling back to standard FSDP/HSDP"
            )
            federated = None
        else:
            federated = tuple(
                federated[0] for _ in range(state.world_size // federated[0])
            )

    if federated is not None:
        if sum(federated) != state.world_size:
            logger.error(
                f"The world size {state.world_size} is not divisible by the specified federated group size {federated} - falling back to standard FSDP/HSDP"
            )
        state.set_federated_scaling(federated_group_size=federated, hsdp=hsdp)
        state.set_exec_device(
            current_device=_get_current_device(state=state), streams=streams
        )
    else:  # Setting non-federated techniques
        if hsdp is not None:
            state.set_hsdp(hsdp=hsdp)
        else:
            state.set_fsdp()

    # Setting initialization device
    init_device, meta_initialization = _get_init_device(state=state)
    state.set_init_device(
        init_device=init_device, meta_initialization=meta_initialization
    )

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
        device_id=state.current_device,
    )


def cleanup_distributed_process_group(state: DistributedState) -> None:
    """Cleanup PyTorch's distributed environment

    To be called AFTER the various processes have completed their work and by ALL processes

    :param state: Instantiated distributed state
    :type state: DistributedState
    """
    logger.debug(f"[Rank {state.rank}]: calling destroy_process_group")

    dist.barrier(group=dist.GroupMember.WORLD, device_ids=[state.node_local_rank])
    dist.destroy_process_group(group=dist.GroupMember.WORLD)


def _get_current_device(
    state: DistributedState,
) -> torch.device | int:
    """PyTorch current device setup

    Returns the device for the current process and empties its cache if it is a GPU

    :param state: Instantiated distributed state
    :type state: DistributedState
    :return: The computation device
    :rtype: torch.device | int
    """

    current_device: torch.device | int = torch.device("cpu")
    if torch.cuda.is_available():
        current_device = (
            torch.device("cuda", state.node_local_rank)
            if state.is_node_setup()
            else torch.device("cuda")
        )
        torch.cuda.set_device(current_device)
        torch.cuda.empty_cache()

    return current_device


def _get_init_device(
    state: DistributedState,
) -> tuple[
    torch.device,
    bool,
]:
    """PyTorch initiazation device setup

    Returns the best initialisation device to load large model in a distributed way with low RAM usage (in case of "meta" model FSDP initialisation should provide sync_module_states=True) and if meta initialisation is required

    :param state: Instantiated distributed state
    :type state: DistributedState
    :return: The device for model initialisation and if meta initialisation is enabled
    :rtype: tuple[torch.device, bool]
    """

    init_device: torch.device = torch.device("cpu")
    meta_initialization: bool = False

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            if (
                (
                    state.is_federated_scaling_setup()
                    and state.is_fsdp_setup()
                    and state.federated_local_rank != 0
                )
                or (
                    not state.is_federated_scaling_setup()
                    and state.is_fsdp_setup()
                    and state.rank != 0
                )
                or (state.is_hsdp_setup() and state.replica_local_rank != 0)
            ):
                init_device = torch.device("meta")
            meta_initialization = True

    return init_device, meta_initialization


def _setup_devices(
    state: DistributedState,
) -> tuple[
    torch.device | int,
    torch.device,
    bool,
]:
    """PyTorch device setup

        Sets the GPU for the current process and empties its cache
        Also, returns the best initialisation device to load large model in a distributed way with low RAM usage (in case of "meta" model FSDP initialisation should provide sync_module_states=True)
    s
        :param state: Instantiated distributed state
        :type state: DistributedState
        :return: Two devices, one for computation and the other for model initialisation, and if meta initialisation is enabled
        :rtype: tuple[int, torch.device | int, bool]
    """

    current_device: torch.device | int = torch.device("cpu")
    if torch.cuda.is_available():
        current_device = (
            torch.device("cuda", state.node_local_rank)
            if state.is_node_setup()
            else torch.device("cuda")
        )
        torch.cuda.set_device(current_device)
        torch.cuda.empty_cache()

    init_device: torch.device = torch.device("cpu")
    meta_initialization: bool = False

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            if (
                (
                    state.is_federated_scaling_setup()
                    and state.is_fsdp_setup()
                    and state.federated_local_rank != 0
                )
                or (
                    not state.is_federated_scaling_setup()
                    and state.is_fsdp_setup()
                    and state.rank != 0
                )
                or (state.is_hsdp_setup() and state.replica_local_rank != 0)
            ):
                init_device = torch.device("meta")
            meta_initialization = True

    return current_device, init_device, meta_initialization
