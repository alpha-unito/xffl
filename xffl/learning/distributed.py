"""Common distributed PyTorch utilities"""

import os
from logging import Logger, getLogger
from typing import Optional, Tuple
from datetime import timedelta

import torch.distributed as dist
from torch.distributed import ProcessGroupNCCL

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def setup_distributed_process_group(
    rank: Optional[int] = None,
    local_rank: Optional[int] = None,
    world_size: Optional[int] = None,
    backend: Optional[str] = "nccl",  # TODO: list alternatives ("nccl", "gloo", "mpi")
) -> Tuple[int, int, int]:
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
    :type backend: Optional[str], optional
    :return: rank and local_rank of the calling process and global world size
    :rtype: Tuple[int, int, int]
    """
    rank = int(os.environ.get("RANK")) if not rank else rank
    local_rank = int(os.environ.get("LOCAL_RANK")) if not local_rank else local_rank
    world_size = int(os.environ.get("WORLD_SIZE")) if not world_size else world_size

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

    return rank, local_rank, world_size


def cleanup_distributed_process_group() -> None:
    """Cleanup PyTorch's distributed environment

    To be called AFTER the various processes have completed their work and by ALL processes
    """
    import time

    # dist.barrier()
    rank = dist.get_rank()
    logger.debug(f"[RANK {rank}] calls destroy_process_group")

    dist.destroy_process_group(dist.GroupMember.WORLD)
