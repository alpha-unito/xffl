"""Common distributed PyTorch utilities"""

import os
import time
from datetime import timedelta
from logging import Logger, getLogger
from typing import List, Literal, Optional, Tuple

import torch.distributed as dist
from torch import cuda, nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.nn.functional import all_reduce

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def federated_averaging(
    model: nn.Module, federated_groups: List[ProcessGroup], local_rank: int
):
    """Federated averaging of corresponding model's shards on different hosts

    :param model: Model to train
    :type model: nn.Module
    :param federated_groups: List of process group to run federated scaling on
    :type federated_groups: List[ProcessGroup]
    :param local_rank: Local rank of the calling process
    :type local_rank: int
    """
    logger.debug(f"Averaging weights...")

    start_time = time.perf_counter()
    for param in model.parameters():  # TODO: raggruppare parametri?
        all_reduce(
            tensor=param,
            op=dist.ReduceOp.AVG,  # TODO: only on NCCL?
            group=federated_groups[local_rank],
        )
    logger.debug(f"Averaging time: {(time.perf_counter() - start_time):.2f} seconds")


def get_timeout(timeout: Optional[int] = 60) -> timedelta:
    """Maximum allowed timeout for distributed communnications

    :param timeout: Maximum allowed timeout in seconds, defaults to 60
    :type timeout: Optional[int], optional
    :return: Maximum allowed time delta
    :rtype: timedelta
    """
    return timedelta(seconds=timeout)


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
    local_rank: Optional[int] = None,
    world_size: Optional[int] = None,
    backend: Optional[Literal["nccl", "gloo", "mpi"]] = "nccl",
    hsdp: Optional[int] = 0,
    federated: Optional[bool] = False,
) -> Tuple[int, int, int, Optional[DeviceMesh]]:
    """Setup PyTorch's distributed environment

    To be called AFTER the various processes have been created and by ALL processes
    The distributed randevouz point is determinated by two environmental variable that should be set BEFORE calling this method (same values for all processes):
    MASTER_ADD:  network address of the rendezvous
    MASTER_PORT: network port of the rendezvous

    :param rank: Rank of tha calling process, otherwise obtained from the environment, defaults to None
    :type rank: Optional[int], optional
    :param local_rank: Local rank of tha calling process, otherwise obtained from the environment, defaults to None
    :type local_rank: Optional[int], optional
    :param world_size: Global world size, otherwise obtained from the environment, defaults to None
    :type world_size: Optional[int], optional
    :param backend: Communication backend to be used, defaults to "nccl" (distributed GPU training)
    :type backend: Optional[Literal["nccl", "gloo", "mpi"]], optional
    :param hsdp: Activate Hybrid Sharding Distributed Parallelism with specified replica group size, defaults to 0
    :type hsdp: Optional[int], optional
    :raises AttributeError: If backend is not in nccl, gloo, or mpi
    :return: Rank and local rank of the calling process, global world size, HSDP device mesh, and federated scaling groups
    :rtype: Tuple[int, int, int, Optional[DeviceMesh], Optional[List[ProcessGroup]]]
    """
    rank: int = int(os.environ.get("RANK")) if not rank else rank
    local_rank: int = (
        int(os.environ.get("LOCAL_RANK")) if not local_rank else local_rank
    )
    world_size: int = (
        int(os.environ.get("WORLD_SIZE")) if not world_size else world_size
    )

    # Requires MASTER_ADDR and MASTER_PORT environmental variables to be set # TODO: add check
    logger.debug(
        f"Setting up process with global rank {rank}, local rank {local_rank} and world size {world_size} through {backend}"
    )
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
        pg_options=(
            get_deafult_nccl_process_group_options() if backend == "nccl" else None
        ),
        timeout=get_timeout(),
    )

    # 2D device mesh creation for HSDP - Sharding intra-group, replicating inter-groups
    hsdp_mesh: Optional[DeviceMesh] = (
        setup_hsdp_mesh(rank=rank, world_size=world_size, hsdp=hsdp) if hsdp else None
    )
    # Federated Scaling process groups - Sharding intra-group, replicating inter-groups
    fed_scaling_groups: Optional[List[ProcessGroup]] = (
        setup_federated_scaling_groups(
            world_size=world_size, hsdp=hsdp, backend=backend
        )
        if hsdp_mesh and federated
        else None
    )

    return rank, local_rank, world_size, hsdp_mesh, fed_scaling_groups


def setup_hsdp_mesh(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    hsdp: Optional[int] = 0,
) -> DeviceMesh:
    """Creates a 2D device mesh allowing Hybrid Sharding Data Parallel (HSDP) training

    :param rank: Rank of tha calling process, otherwise obtained from the environment, defaults to None
    :type rank: Optional[int], optional
    :param world_size: Global world size, otherwise obtained from the environment, defaults to None
    :type world_size: Optional[int], optional
    :param hsdp: Activate Hybrid Sharding Distributed Parallelism with specified replica group size, defaults to 0
    :type hsdp: Optional[int], optional
    :return: 2D device mesh suited for HSDP training
    :rtype: DeviceMesh
    """
    group_world_size: int = world_size // hsdp
    if rank == 0:
        logger.debug(
            f"Setting up HSDP device mesh with replica group size {hsdp} and group world size {group_world_size}"
        )

    if group_world_size > 0 and world_size % hsdp == 0:
        mesh: DeviceMesh = init_device_mesh(
            device_type="cuda" if cuda.is_available() else "cpu",
            mesh_shape=[group_world_size, hsdp],
            mesh_dim_names=["replica", "shard"],
        )
        if rank == 0:
            logger.debug(f"Obtained HSDP mesh: {mesh}")
    else:
        if rank == 0:
            logger.error(
                f"Impossible setting up HSDP device mesh with replica group size {hsdp} and group world size {group_world_size}: world size is not divisible by local world size into the requested group number.\nFalling back to FSDP."
            )
    return mesh


def setup_federated_scaling_groups(
    world_size: Optional[int] = None,
    hsdp: Optional[int] = 0,
    backend: Optional[Literal["nccl", "gloo", "mpi"]] = "nccl",
) -> List[ProcessGroup]:
    """Create the federated scaling rank groups

    This process groups bring together all the ranks handling corresponding model's shards.
    E.g.: if a model is sharded among four processes and replicated across two process groups (i.e., device_mesh=[[0,1,2,3],[4,5,6,7]])
    then the federated scaling process groups correspond to the groups of processes having the same local rank (i.e., [[0,4][1,5][2,6][3,7]])

    :param world_size: Global world size, otherwise obtained from the environment, defaults to None
    :type world_size: Optional[int], optional
    :param hsdp: Activate Hybrid Sharding Distributed Parallelism with specified replica group size, defaults to 0
    :type hsdp: Optional[int], optional
    :param backend: Communication backend to be used, defaults to "nccl" (distributed GPU training)
    :type backend: Optional[Literal["nccl", "gloo", "mpi"]], optional
    :return: List of process groups handling the same model's shards
    :rtype: List[ProcessGroup]
    """
    logger.debug(f"Creating federated scaling process groups...")
    federated_groups: List[ProcessGroup] = []
    federated_groups_ranks: List[int] = []
    for group_rank in range(hsdp):
        ranks: List[int] = [
            group_rank + local_rank * hsdp for local_rank in range(world_size // hsdp)
        ]

        federated_groups_ranks.append(ranks)

        federated_groups.append(
            dist.new_group(
                ranks=ranks,
                timeout=get_timeout(),
                backend=backend,
                pg_options=get_deafult_nccl_process_group_options(),
                use_local_synchronization=False,
                group_desc=f"Local ranks {group_rank}",
            )
        )
    logger.debug(f"Created federated scaling process groups: {federated_groups_ranks}")

    return federated_groups


def cleanup_distributed_process_group() -> None:
    """Cleanup PyTorch's distributed environment

    To be called AFTER the various processes have completed their work and by ALL processes
    """
    rank: int = dist.get_rank()
    logger.debug(f"Rank {rank} calls destroy_process_group")

    dist.destroy_process_group(dist.GroupMember.WORLD)
