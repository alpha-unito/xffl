"""Utility methods for distributed training"""

from logging import Logger, getLogger

from torch import distributed as dist

from xffl.distributed.distributed_state import DistributedState

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def is_broadcast_necessary(state: DistributedState) -> bool:
    """Checks if the current rank needs to take part into a weights broadcast

    :param state: xFFL distributed state (rank, world_size, backend)
    :type state: DistributedState
    """
    return (
        state.replica_group is not None
        and len(dist.get_process_group_ranks(state.replica_group[0])) > 1
    )
