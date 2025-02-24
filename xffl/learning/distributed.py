"""Common distributed PyTorch utilities"""

import os
from datetime import timedelta
from logging import Logger, getLogger
from typing import Literal, Optional, Tuple

import torch.distributed as dist
from torch.distributed import ProcessGroupNCCL
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def setup_distributed_process_group(
    rank: Optional[int] = None,
    local_rank: Optional[int] = None,
    world_size: Optional[int] = None,
    local_world_size: Optional[int] = None,
    group_world_size: Optional[int] = None,
    backend: Optional[Literal["nccl", "gloo", "mpi"]] = "nccl",
    hsdp: Optional[int] = 0,
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
    :param local_world_size: Local world size, otherwise obtained from the environment, defaults to None
    :type local_world_size: Optional[int], optional
    :param group_world_size: Group world size, otherwise obtained from the environment, defaults to None
    :type group_world_size: Optional[int], optional
    :param backend: Communication backend to be used, defaults to "nccl" (distributed GPU training)
    :type backend: Optional[Literal["nccl", "gloo", "mpi"]], optional
    :param hsdp: Activate Hybrid Sharding Distributed Parallelism with specified replica group size, defaults to 0
    :type hsdp: Optional[int], optional
    :raises AttributeError: If backend is not in nccl, gloo, or mpi
    :return: Rank and local rank of the calling process and global world size
    :rtype: Tuple[int, int, int, Optional[DeviceMesh]]
    """

    if backend not in ["nccl", "gloo", "mpi"]:
        logger.error(f"Specified backend not in [nccl, gloo, mpi]")
        raise AttributeError(f"Specified backend not in [nccl, gloo, mpi]")

    rank: int = int(os.environ.get("RANK")) if not rank else rank
    local_rank: int = (
        int(os.environ.get("LOCAL_RANK")) if not local_rank else local_rank
    )
    world_size: int = (
        int(os.environ.get("WORLD_SIZE")) if not world_size else world_size
    )

    options = ProcessGroupNCCL.Options()
    options.is_high_priority_stream = True
    options._timeout = timedelta(seconds=60)

    # Requires MASTER_ADDR and MASTER_PORT environmental variables to be set
    logger.debug(
        f"Setting up process with global rank {rank}, local rank {local_rank} and world size {world_size} through {backend}"
    )
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
        pg_options=options,
        timeout=timedelta(seconds=60),
    )

    # 2D device mesh creation for HSDP - Sharding intra-group, replicating inter-groups
    mesh = None
    if hsdp:
        local_world_size: int = (
            int(os.environ.get("LOCAL_WORLD_SIZE"))
            if not local_world_size
            else local_world_size
        )
        if local_world_size != hsdp:
            logger.warning(
                f"Specified HSDP replica group size {hsdp} differs from retrieved local world size {local_world_size}"
            )

        group_world_size: int = (
            int(os.environ.get("GROUP_WORLD_SIZE"))
            if not group_world_size
            else group_world_size
        )

        if hsdp * group_world_size == world_size:
            logger.debug(
                f"Setting up HSDP device mesh with local world size {local_world_size} and group world size {group_world_size}"
            )
            mesh: DeviceMesh = init_device_mesh(
                device_type="cuda",
                mesh_shape=[group_world_size, local_world_size],
                mesh_dim_names=["replica", "shard"],
            )
            logger.debug(f"Obtained HSDP mesh: {mesh}")
        else:
            logger.error(
                f"Impossible setting up HSDP device mesh with local world size {hsdp} and group world size {group_world_size}: world size is not divisible by local world size into the requested group number."
            )

    return rank, local_rank, world_size, mesh


def cleanup_distributed_process_group() -> None:
    """Cleanup PyTorch's distributed environment

    To be called AFTER the various processes have completed their work and by ALL processes
    """
    import time

    # dist.barrier()
    rank = dist.get_rank()
    logger.debug(f"[RANK {rank}] calls destroy_process_group")

    dist.destroy_process_group(dist.GroupMember.WORLD)
