"""Utility methods for distributed training"""

from contextlib import nullcontext
from dataclasses import dataclass
from logging import Logger, getLogger
from typing import ContextManager, Tuple

import torch
from torch import cuda
from torch import distributed as dist

from xffl.distributed.distributed_state import DistributedState

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


@dataclass
class Strategy:
    """Aggregation strategy class

    Each strategy is described as a mapping in which each element corresponds to a single communication

    :param mapping: Mapping between original layers index, layer parameter, appropriate context manager for communication, and CUDA stream index
    :type mapping: Tuple[Tuple[Tuple[int, ...], torch.Tensor, ContextManager, int], ...]
    :param reduce_op: All reduce operation
    :type reduce_op: dist.ReduceOp.RedOpType
    :param use_contiguous_memory: Convert tensors to contiguous memory before communication
    :type use_contiguous_memory: bool
    :param src: Source rank for broadcast communications
    :type src: int
    :param broadcast: If it is necessary to run broadcasts
    :type broadcast: bool
    :param state: xFFL distributed state
    :type state: DistributedState
    :param requires_copy: Specified strategy requires copying back the aggregated tensors, defaults to False
    :type requires_copy: bool
    :param param_list: Original model parameter list necessary for copying, defaults to None
    :type param_list: Optional[List[torch.Tensor]]
    """

    mapping: Tuple[Tuple[Tuple[int, ...], torch.Tensor, ContextManager, int], ...]
    reduce_op: dist.ReduceOp.RedOpType
    use_contiguous_memory: bool
    src: int
    broadcast: bool


def _is_broadcast_necessary(state: DistributedState) -> bool:
    """Checks if the current rank needs to take part into a weights broadcast

    :param state: xFFL distributed state (rank, world_size, backend)
    :type state: DistributedState
    """
    return (
        state.replica_group is not None
        and len(dist.get_process_group_ranks(state.replica_group[0])) > 1
    )


def _setup_streams(
    use_multiple_cuda_streams: bool, state: DistributedState
) -> Tuple[int, int, dist.ReduceOp.RedOpType, ContextManager]:
    """Sets up the CUDA streams infos and reduce operation for the aggregation

    :param use_multiple_cuda_streams: If multiple CUDA streams should be used, defaults to False
    :type use_multiple_cuda_streams: bool
    :param state: The xFFL distributed state
    :type state: DistributedState
    :return: The number of streams, the index of the current stream, the reduce operation type, and the context manager for the stream
    :rtype: Tuple[int, int, dist.ReduceOp.RedOpType, ContextManager]
    """
    stream_number: int = 1
    if use_multiple_cuda_streams:
        if state.streams is None or len(state.streams) < 2:
            logger.warning(
                f"Impossible using multiple CUDA streams: current CUDA stream(s) -> {state.streams}"
            )
            use_multiple_cuda_streams = False
        else:
            stream_number = len(state.streams)

    stream_index: int = 0
    reduce_op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM
    stream_context: ContextManager = nullcontext
    if state.backend == "nccl":
        reduce_op = dist.ReduceOp.AVG
        stream_context = torch.cuda.StreamContext(cuda.default_stream())

    return stream_number, stream_index, reduce_op, stream_context


def _all_reduce(
    strategy: Strategy, state: DistributedState
) -> None:  # TODO: torch profiler?
    """AllReduce part of the aggregation - benchmark purpose only

    Executes an all-reduce operation to average the model's weights according to the specified strategy and current distributed state configuration

    :param strategy: Aggregation strategy
    :type strategy: Strategy
    :param state: xFFL distributed state
    :type state: DistributedState
    """
    for _, tensor, stream_context, stream_index in strategy.mapping:
        with stream_context:
            if isinstance(tensor, list):
                dist.all_reduce_coalesced(
                    tensors=tensor,
                    op=strategy.reduce_op,
                    group=state.federated_group[stream_index],
                )  # Coalesced all-reduce for lists of tensors will be deprecated in future versions of PyTorch
            else:
                dist.all_reduce(
                    tensor=tensor,
                    op=strategy.reduce_op,
                    group=state.federated_group[stream_index],
                )


def _all_reduce_and_broadcast(strategy: Strategy, state: DistributedState) -> None:
    """Communication part of the aggregation - complementary to the selected strategy

    Executes an all-reduce followed by a broadcast (if needed) to average the model's weights according to the specified strategy and current distributed state configuration

    :param strategy: Aggregation strategy
    :type strategy: Strategy
    :param state: xFFL distributed state
    :type state: DistributedState
    """
    index: int = 0
    for _, tensor, stream_context, stream_index in strategy.mapping:
        with stream_context:
            dist.all_reduce(
                tensor=tensor,
                op=strategy.reduce_op,
                group=state.federated_group[stream_index],
            )

            if strategy.reduce_op == dist.ReduceOp.SUM:
                torch.Tensor.div_(tensor, state.federated_world_size)

            if strategy.broadcast:
                dist.broadcast(
                    tensor=tensor,
                    src=strategy.src,
                    group=state.replica_group[stream_index],
                )

            # TODO: Rimettere a posto
