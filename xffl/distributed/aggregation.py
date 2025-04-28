"""Aggregation strategies for local xFFL"""

import itertools
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from logging import Logger, getLogger
from typing import ContextManager, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import cuda, nn

from xffl.distributed.distributed_state import DistributedState
from xffl.distributed.utils import is_broadcast_necessary
from xffl.learning.utils import get_model_size, get_model_size_in_bits

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


@dataclass
class Strategy:
    mapping: Tuple[Tuple[Tuple[int, ...], torch.Tensor, ContextManager, int], ...]
    reduce_op: dist.ReduceOp.RedOpType
    use_contiguous_memory: bool
    src: int
    broadcast: bool
    requires_copy: bool = False
    param_list: Optional[List[torch.Tensor]] = None


def benchmark_aggregation_strategies(
    state: DistributedState, model: nn.Module, iterations: int = 10
) -> None:
    aggregation_strategies: Tuple[callable, ...] = (
        layer_by_layer,
        layer_by_layer_optimized,
        stacked,
        stacked_optimized,
    )

    for aggregation_strategy in aggregation_strategies:
        if state.rank == 0:
            print("\n")

        for use_multiple_cuda_streams, use_contiguous_memory in itertools.product(
            [False, True], repeat=2
        ):

            strategy: Strategy = aggregation_strategy(
                model=model,
                state=state,
                use_multiple_cuda_streams=use_multiple_cuda_streams,
                use_contiguous_memory=use_contiguous_memory,
            )

            # if state.rank == 0:
            #    logger.debug(strategy)

            start_time = time.perf_counter()
            for _ in range(iterations):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                all_reduce_and_broadcast(strategy=strategy, state=state)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            comm_time = (time.perf_counter() - start_time) / iterations

            if state.rank == 0:
                logger.debug(
                    f"{aggregation_strategy.__name__} - Multiple CUDA streams {use_multiple_cuda_streams}, Contiguous memory {use_contiguous_memory}: {comm_time:.2f}"
                )
                if not is_broadcast_necessary(state=state):
                    throughput: float = (
                        (get_model_size_in_bits(model=model) / comm_time)
                        * (
                            2
                            * (state.federated_world_size - 1)
                            / state.federated_world_size
                        )
                        / 10**9
                    )  # Based on https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
                    logger.debug(f"AllReduce throughput: {throughput:.2f} Gb/s")

    if state.rank == 0:
        print("\n")


def all_reduce_and_broadcast(strategy: Strategy, state: DistributedState) -> None:
    for layer_index, tensor, stream_context, stream_index in strategy.mapping:
        with stream_context:
            dist.all_reduce(
                tensor=(
                    tensor.contiguous() if strategy.use_contiguous_memory else tensor
                ),
                op=strategy.reduce_op,
                group=state.federated_group[stream_index],
            )

            if strategy.reduce_op == dist.ReduceOp.SUM:
                torch.Tensor.div_(tensor, state.federated_world_size)

            if strategy.broadcast:
                dist.broadcast(
                    tensor=(
                        tensor.contiguous()
                        if strategy.use_contiguous_memory
                        else tensor
                    ),
                    src=strategy.src,
                    group=state.replica_group[stream_index],
                )

            if strategy.requires_copy:
                for aggregated_index, local_index in enumerate(layer_index):
                    strategy.param_list[local_index].copy_(
                        tensor[aggregated_index], non_blocking=True
                    )


def layer_by_layer(
    model: nn.Module,
    state: DistributedState,
    use_multiple_cuda_streams: bool = False,
    use_contiguous_memory: bool = False,
) -> Strategy:
    """Layer-by-layer aggregation

    In case of multiple CUDA streams the layers are assigned to each of them in a round-robin fashion

    :param model: PyTorch model
    :type model: nn.Module
    :param state: xFFL distributed state
    :type state: DistributedState
    :param use_multiple_cuda_streams: use multiple CUDA streams if available, defaults to False
    :type use_multiple_cuda_streams: bool
    :param use_contiguous_memory: convert tensors to a contiguous memory representation, defaults to False
    :type use_contiguous_memory: bool
    :returns: The Aggregation strategy configuration
    :rtype: Strategy
    """
    if use_multiple_cuda_streams:
        if state.streams is None or len(state.streams) < 2:
            logger.warning(
                f"Impossible using multiple CUDA streams: current CUDA stream(s) -> {state.streams}"
            )
            use_multiple_cuda_streams = False

    stream_index: int = 0
    reduce_op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM
    stream_context: ContextManager = nullcontext
    if state.backend == "nccl":
        reduce_op = dist.ReduceOp.AVG
        stream_context = torch.cuda.StreamContext(cuda.default_stream())

    mapping: List[Tuple[Tuple[int, ...], torch.Tensor, ContextManager, int]] = []
    for layer_index, param in enumerate(model.parameters()):
        if use_multiple_cuda_streams:
            stream_index = layer_index % len(state.streams)
            stream_context = torch.cuda.StreamContext(state.streams[stream_index])
        mapping.append(
            (
                (layer_index,),
                param,
                stream_context,
                stream_index,
            )
        )

    return Strategy(
        mapping=tuple(mapping),
        reduce_op=reduce_op,
        use_contiguous_memory=use_contiguous_memory,
        src=state.rank if state.is_sender else state.receive_from,
        broadcast=is_broadcast_necessary(state=state),
    )


def layer_by_layer_optimized(
    model: nn.Module,
    state: DistributedState,
    use_multiple_cuda_streams: bool = False,
    use_contiguous_memory: bool = False,
) -> Strategy:
    """Layer-by-layer aggregation through a bucket-based approach

    In case of multiple CUDA streams the layers are assigned to each of them trying to divide the parameters equally

    :param model: PyTorch model
    :type model: nn.Module
    :param state: xFFL distributed state
    :type state: DistributedState
    :param use_multiple_cuda_streams: use multiple CUDA streams if available, defaults to False
    :type use_multiple_cuda_streams: bool
    :param use_contiguous_memory: convert tensors to a contiguous memory representation, defaults to False
    :type use_contiguous_memory: bool
    :returns: The Aggregation strategy configuration
    :rtype: Strategy
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

    param_list: List[torch.Tensor] = list(model.parameters())
    bucket_size: int = get_model_size(model=model) // stream_number

    parameter_counter: int = 0
    mapping: List[Tuple[Tuple[int, ...], torch.Tensor, ContextManager, int]] = []
    for layer_index, layer in enumerate(param_list):
        if use_multiple_cuda_streams:
            stream_context = torch.cuda.StreamContext(state.streams[stream_index])

        mapping.append(
            (
                (layer_index,),
                layer,
                stream_context,
                stream_index,
            )
        )
        parameter_counter += layer.numel()
        if parameter_counter >= bucket_size:
            stream_index += 1
            parameter_counter = 0

    return Strategy(
        mapping=tuple(mapping),
        reduce_op=reduce_op,
        use_contiguous_memory=use_contiguous_memory,
        src=state.rank if state.is_sender else state.receive_from,
        broadcast=is_broadcast_necessary(state=state),
    )


def stacked(
    model: nn.Module,
    state: DistributedState,
    use_multiple_cuda_streams: bool = False,
    use_contiguous_memory: bool = False,
) -> Strategy:
    """Weights averaging through a stacking-based approach

    All layers are stacked into "buckets" according to their size and sent on different CUDA streams in a round-robin fashion

    :param model: PyTorch model
    :type model: nn.Module
    :param state: xFFL distributed state
    :type state: DistributedState
    :param use_multiple_cuda_streams: use multiple CUDA streams if available, defaults to False
    :type use_multiple_cuda_streams: bool
    :param use_contiguous_memory: convert tensors to a contiguous memory representation, defaults to False
    :type use_contiguous_memory: bool
    :returns: The Aggregation strategy configuration
    :rtype: Strategy
    """
    if use_multiple_cuda_streams:
        if state.streams is None or len(state.streams) < 2:
            logger.warning(
                f"Impossible using multiple CUDA streams: current CUDA stream(s) -> {state.streams}"
            )
            use_multiple_cuda_streams = False

    stream_index: int = 0
    reduce_op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM
    stream_context: ContextManager = nullcontext
    if state.backend == "nccl":
        reduce_op = dist.ReduceOp.AVG
        stream_context = torch.cuda.StreamContext(cuda.default_stream())

    param_list: List[torch.Tensor] = list(model.parameters())

    size_map: Dict[int, List[int]] = defaultdict(list)
    for layer_index, layer in enumerate(param_list):
        size_map[layer.numel()].append(layer_index)

    mapping: List[Tuple[Tuple[int, ...], torch.Tensor, ContextManager, int]] = []
    for index, (_, layer_list) in enumerate(size_map.items()):
        if use_multiple_cuda_streams:
            stream_index = index % len(state.streams)
            stream_context = torch.cuda.StreamContext(state.streams[stream_index])
        mapping.append(
            (
                tuple(layer_list),
                torch.stack([param_list[index] for index in layer_list]),
                stream_context,
                stream_index,
            )
        )

    return Strategy(
        mapping=tuple(mapping),
        reduce_op=reduce_op,
        use_contiguous_memory=use_contiguous_memory,
        src=state.rank if state.is_sender else state.receive_from,
        broadcast=is_broadcast_necessary(state=state),
        requires_copy=True,
        param_list=param_list,
    )


def stacked_optimized(
    model: nn.Module,
    state: DistributedState,
    use_multiple_cuda_streams: bool = False,
    use_contiguous_memory: bool = False,
) -> Strategy:
    """Weights averaging through a stacking-based approach

    All layers are stacked into "buckets" according to their size and sent on different CUDA streams trying to divide the information sent equally between them

    :param model: PyTorch model
    :type model: nn.Module
    :param state: xFFL distributed state
    :type state: DistributedState
    :param use_multiple_cuda_streams: use multiple CUDA streams if available, defaults to False
    :type use_multiple_cuda_streams: bool
    :param use_contiguous_memory: convert tensors to a contiguous memory representation, defaults to False
    :type use_contiguous_memory: bool
    :returns: The Aggregation strategy configuration
    :rtype: Strategy
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

    reduce_op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM
    stream_context: ContextManager = nullcontext
    if state.backend == "nccl":
        reduce_op = dist.ReduceOp.AVG
        stream_context = torch.cuda.StreamContext(cuda.default_stream())

    param_list: List[torch.Tensor] = list(model.parameters())
    bucket_size: int = get_model_size(model=model) // stream_number

    size_map: Dict[int, List[int]] = defaultdict(list)
    for layer_index, layer in enumerate(param_list):
        size_map[layer.numel()].append(layer_index)

    stream_index: int = 0
    parameter_counter: int = 0
    layers_in_current_stack: List[int] = []
    mapping: List[Tuple[Tuple[int, ...], torch.Tensor, ContextManager, int]] = []
    for layer_size, layer_list in size_map.items():
        for i, layer in enumerate(layer_list):
            if use_multiple_cuda_streams:
                stream_context = torch.cuda.StreamContext(state.streams[stream_index])

            parameter_counter += layer_size
            layers_in_current_stack.append(layer)
            if i == len(layer_list) - 1 or parameter_counter >= bucket_size:
                mapping.append(
                    (
                        tuple(layers_in_current_stack),
                        torch.stack(
                            [param_list[index] for index in layers_in_current_stack]
                        ),
                        stream_context,
                        stream_index,
                    )
                )
                layers_in_current_stack = []
            if parameter_counter >= bucket_size:
                stream_index += 1
                parameter_counter = 0

    return Strategy(
        mapping=tuple(mapping),
        reduce_op=reduce_op,
        use_contiguous_memory=use_contiguous_memory,
        src=state.rank if state.is_sender else state.receive_from,
        broadcast=is_broadcast_necessary(state=state),
        requires_copy=True,
        param_list=param_list,
    )
