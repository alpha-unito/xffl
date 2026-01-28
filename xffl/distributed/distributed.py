"""Common distributed PyTorch utilities"""

import os
from logging import Logger, getLogger
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import Backend
from torch.distributed.fsdp import ShardingStrategy

from xffl.custom.config import XFFLConfig
from xffl.distributed.distributed_state import DistributedState
from xffl.utils.utils import (
    get_default_nccl_process_group_options,
    get_timeout,
    resolve_param,
)

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def _get_int_from_env(var: Optional[int], env_var: str) -> Optional[int]:
    if var is None:
        _var: Optional[str] = os.environ.get(env_var)
        if _var is not None:
            var = int(_var)
    return var


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
    federated: Optional[int | Tuple[int, ...]] = None,
    streams: Optional[int] = None,
    config: Optional[XFFLConfig] = None,
) -> DistributedState:
    """Setup PyTorch's distributed environment

    To be called AFTER the various processes have been created and by ALL processes
    The distributed rendez-vous point determined by two environmental variable that should be set BEFORE calling this method (same values for all processes):
    MASTER_ADD:  network address of the rendezvous
    MASTER_PORT: network port of the rendezvous

    The parameters can be provided both directly and through an XFFL configuration.
    In case both are provided, the firsts take the precedence.

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
    :param federated: Activate Federated Scaling with specified federated group size, defaults to None
    :type federated: Optional[int | Tuple[int, ...]], optional
    :param streams: Number of CUDA streams to instantiate, defaults to None
    :type streams: Optional[int]
    :param config: XFFL configuration
    :type config: Optional[XFFLConfig], defaults to None
    :raises AttributeError: If backend is not in nccl, gloo, or mpi
    :raises ValueError: If no valid MASTER_ADDR and MASTER_PORT are set
    :return: Distributed state of the current training setup
    :rtype: DistributedState
    """

    # Parameters resolution
    _rank: Optional[int] = resolve_param(value=rank, config=config, attr="rank")
    _world_size: Optional[int] = resolve_param(
        value=world_size, config=config, attr="world_size"
    )
    _group_local_rank: Optional[int] = resolve_param(
        value=group_local_rank, config=config, attr="group_local_rank"
    )
    _group_local_size: Optional[int] = resolve_param(
        value=group_local_size, config=config, attr="group_local_size"
    )
    _group_rank: Optional[int] = resolve_param(
        value=group_rank, config=config, attr="group_rank"
    )
    _group_world_size: Optional[int] = resolve_param(
        value=group_world_size, config=config, attr="group_world_size"
    )
    _backend: Optional[Backend] = resolve_param(
        value=backend, config=config, attr="backend"
    )
    _master_addr: Optional[str] = resolve_param(
        value=master_addr, config=config, attr="master_addr"
    )
    _master_port: Optional[int] = resolve_param(
        value=master_port, config=config, attr="master_port"
    )
    _device: Optional[torch.device] = resolve_param(
        value=device, config=config, attr="device"
    )
    _hsdp: Optional[int] = resolve_param(value=hsdp, config=config, attr="hsdp")
    __federated: Optional[int | Tuple[int, ...]] = resolve_param(
        value=federated, config=config, attr="federated"
    )
    _federated: Optional[Tuple[int, ...]] = (
        (__federated,) if isinstance(__federated, int) else __federated
    )
    _streams: Optional[int] = resolve_param(
        value=streams, config=config, attr="streams"
    )

    state: DistributedState = DistributedState()

    # Distributed state global information
    state.set_global(
        backend=(
            _backend
            if _backend is not None
            else "nccl" if torch.cuda.is_available() else "gloo"
        ),
        device_type=(
            _device
            if _device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ),
        master_addr=(
            os.environ.get("MASTER_ADDR") if _master_addr is None else _master_addr
        ),
        master_port=_get_int_from_env(var=_master_port, env_var="MASTER_PORT"),
        rank=_get_int_from_env(var=_rank, env_var="RANK"),
        world_size=_get_int_from_env(var=_world_size, env_var="WORLD_SIZE"),
    )

    # Distributed state local information
    state.set_node(
        node_local_rank=_get_int_from_env(var=_group_local_rank, env_var="LOCAL_RANK"),
        node_local_size=_get_int_from_env(
            var=_group_local_size, env_var="LOCAL_WORLD_SIZE"
        ),
        node_rank=_get_int_from_env(var=_group_rank, env_var="GROUP_RANK"),
        node_world_size=_get_int_from_env(
            var=_group_world_size, env_var="GROUP_WORLD_SIZE"
        ),
    )

    # Check if Federated Scaling is required
    assert state.node_local_size is not None
    assert state.world_size is not None

    if _federated is None:
        if "XFFL_FEDERATED_LOCAL_WORLD_SIZE" in os.environ:
            _federated = tuple(
                int(item) * state.node_local_size
                for item in str(
                    os.environ.get("XFFL_FEDERATED_LOCAL_WORLD_SIZE")
                ).split(",")
            )
    elif len(_federated) == 1:
        if state.world_size % _federated[0] != 0:
            logger.error(
                f"The world size {state.world_size} is not divisible by the specified federated group size {_federated[0]} - falling back to standard FSDP/HSDP"
            )
            _federated = None
        elif state.world_size == _federated[0]:
            logger.error(
                f"The world size {state.world_size} and the the specified federated group size {_federated[0]} are equal - falling back to standard FSDP/HSDP"
            )
            _federated = None
        else:
            _federated = tuple(
                _federated[0] for _ in range(state.world_size // _federated[0])
            )

    # Setting execution device
    state.set_exec_device(
        current_device=_get_current_device(state=state),
        streams=(_streams if _federated is not None else None),
    )

    # Basic PyTorch distributed setup
    init_distributed_process_group(state=state)

    if _federated is not None:
        if sum(_federated) != state.world_size:
            logger.error(
                f"The world size {state.world_size} is not divisible by the specified federated group size {_federated} - falling back to standard FSDP/HSDP"
            )
        state.set_federated_scaling(federated_group_size=_federated, hsdp=_hsdp)
    else:  # Setting non-federated techniques
        if _hsdp is not None:
            state.set_hsdp(hsdp=_hsdp)
        else:
            state.set_fsdp()

    # Setting initialization device
    init_device, meta_initialization = _get_init_device(state=state)
    state.set_init_device(
        init_device=init_device, meta_initialization=meta_initialization
    )

    logger.debug(f"{state}")
    return state


def init_distributed_process_group(state: DistributedState) -> None:
    """PyTorch's distributed backend initialization

    :param state: Partially instantiated distributed state (rank, world_size, backend)
    :type state: DistributedState
    """
    if state.world_size is None or state.rank is None:
        logger.error(
            f"Impossible setting up the distributed process group: the world size {state.world_size} and/or the rank {state.rank} are not correctly setup."
        )
    else:
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
            # device_id=state.current_device,  # TODO: this does not seems to work properly with device meshes
        )


def cleanup_distributed_process_group(
    state: DistributedState, del_obj: Optional[Tuple[Any, ...]] = None
) -> None:
    """Cleanup PyTorch's distributed environment

    To be called AFTER the various processes have completed their work and by ALL processes

    :param state: Instantiated distributed state
    :type state: DistributedState
    :param del_obj: Objects to be deleted before destroying the process group, defaults to []
    :type del_obj: Tuple[Any]
    """
    if del_obj is not None:
        for obj in del_obj:
            del obj

    if dist.is_initialized():
        dist.barrier(device_ids=[state.node_local_rank])
        dist.destroy_process_group()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _get_current_device(
    state: DistributedState,
) -> torch.device:
    """PyTorch current device setup

    Returns the device for the current process and empties its cache if it is a GPU

    :param state: Instantiated distributed state
    :type state: DistributedState
    :return: The computation device
    :rtype: torch.device | int
    """
    assert state.node_local_rank is not None

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
