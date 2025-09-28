"""Aggregation strategies for local xFFL"""

from logging import Logger, getLogger
from typing import ContextManager, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from xffl.distributed.distributed_state import DistributedState
from xffl.distributed.utils import (
    Strategy,
    _all_reduce,
    _is_broadcast_necessary,
    _setup_streams,
)
from xffl.learning.utils import get_model_size, get_model_size_in_bits

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


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
    stream_number, stream_index, reduce_op, stream_context = _setup_streams(
        use_multiple_cuda_streams=use_multiple_cuda_streams, state=state
    )

    mapping: List[Tuple[Tuple[int, ...], torch.Tensor, ContextManager, int]] = []
    for layer_index, layer in enumerate(model.parameters()):
        if use_multiple_cuda_streams:
            stream_index = layer_index % stream_number
            stream_context = torch.cuda.StreamContext(state.streams[stream_index])
        with stream_context:
            mapping.append(
                (
                    (layer_index,),
                    layer.contiguous() if use_contiguous_memory else layer,
                    stream_context,
                    stream_index,
                )
            )

    return Strategy(
        mapping=tuple(mapping),
        reduce_op=reduce_op,
        use_contiguous_memory=use_contiguous_memory,
        src=state.rank if state.is_sender else state.receive_from,
        broadcast=_is_broadcast_necessary(state=state),
    )


def layer_by_layer_(
    model: nn.Module,
    state: DistributedState,
    use_multiple_cuda_streams: bool = False,
    use_contiguous_memory: bool = False,
) -> None:
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
    """
    stream_number, stream_index, reduce_op, stream_context = _setup_streams(
        use_multiple_cuda_streams=use_multiple_cuda_streams, state=state
    )

    for layer_index, param in enumerate(model.parameters()):
        if use_multiple_cuda_streams:
            stream_index = layer_index % stream_number
            stream_context = torch.cuda.StreamContext(state.streams[stream_index])
        with stream_context:
            dist.all_reduce(
                tensor=param.contiguous() if use_contiguous_memory else param,
                op=reduce_op,
                group=state.federated_group[stream_index],
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
    stream_number, stream_index, reduce_op, stream_context = _setup_streams(
        use_multiple_cuda_streams=use_multiple_cuda_streams, state=state
    )

    bucket_size: int = get_model_size(model=model) // stream_number

    parameter_counter: int = 0
    mapping: List[Tuple[Tuple[int, ...], torch.Tensor, ContextManager, int]] = []
    for layer_index, layer in enumerate(model.parameters()):
        if use_multiple_cuda_streams:
            stream_context = torch.cuda.StreamContext(state.streams[stream_index])
        with stream_context:
            mapping.append(
                (
                    (layer_index,),
                    layer.contiguous() if use_contiguous_memory else layer,
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
        broadcast=_is_broadcast_necessary(state=state),
    )


def layer_by_layer_optimized_(
    model: nn.Module,
    state: DistributedState,
    use_multiple_cuda_streams: bool = False,
    use_contiguous_memory: bool = False,
) -> None:
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
    """
    stream_number, stream_index, reduce_op, stream_context = _setup_streams(
        use_multiple_cuda_streams=use_multiple_cuda_streams, state=state
    )

    bucket_size: int = get_model_size(model=model) // stream_number

    parameter_counter: int = 0
    for layer in model.parameters():
        if use_multiple_cuda_streams:
            stream_context = torch.cuda.StreamContext(state.streams[stream_index])
        with stream_context:
            dist.all_reduce(
                tensor=layer.contiguous() if use_contiguous_memory else layer,
                op=reduce_op,
                group=state.federated_group[stream_index],
            )
            parameter_counter += layer.numel()
            if parameter_counter >= bucket_size:
                stream_index += 1
                parameter_counter = 0


def bucket_flatten(
    model: nn.Module,
    state: DistributedState,
    use_multiple_cuda_streams: bool = False,
    use_contiguous_memory: bool = False,
) -> Strategy:
    """Weights averaging through a bucket-based approach

    All layers are stacked into "buckets" and sent on different CUDA streams in a round-robin fashion

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
    stream_number, stream_index, reduce_op, stream_context = _setup_streams(
        use_multiple_cuda_streams=use_multiple_cuda_streams, state=state
    )

    param_list: List[torch.Tensor] = list(model.parameters())

    buckets: List[List[torch.Tensor]] = [[] for _ in range(stream_number)]
    for layer_index, _ in enumerate(model.parameters()):
        buckets[layer_index % stream_number].append(layer_index)

    mapping: List[Tuple[Tuple[int, ...], torch.Tensor, ContextManager, int]] = []
    for stream_index, bucket in enumerate(buckets):
        if use_multiple_cuda_streams:
            stream_context = torch.cuda.StreamContext(state.streams[stream_index])
        with stream_context:
            flatten_bucket: torch.Tensor = _flatten_dense_tensors(
                [param_list[i] for i in bucket]
            )
            mapping.append(
                (
                    tuple(bucket),
                    (
                        flatten_bucket.contiguous()
                        if use_contiguous_memory
                        else flatten_bucket
                    ),
                    stream_context,
                    stream_index,
                )
            )

    return Strategy(
        mapping=tuple(mapping),
        reduce_op=reduce_op,
        use_contiguous_memory=use_contiguous_memory,
        src=state.rank if state.is_sender else state.receive_from,
        broadcast=_is_broadcast_necessary(state=state),
    )


def bucket_coalesced(
    model: nn.Module,
    state: DistributedState,
    use_multiple_cuda_streams: bool = False,
    use_contiguous_memory: bool = False,
) -> Strategy:
    """Weights averaging through a bucket-based approach

    All layers are stacked into "buckets" and sent on different CUDA streams in a round-robin fashion

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
    stream_number, stream_index, reduce_op, stream_context = _setup_streams(
        use_multiple_cuda_streams=use_multiple_cuda_streams, state=state
    )

    param_list: List[torch.Tensor] = list(model.parameters())

    buckets: List[List[torch.Tensor]] = [[] for _ in range(stream_number)]
    for layer_index, _ in enumerate(model.parameters()):
        buckets[layer_index % stream_number].append(layer_index)

    mapping: List[Tuple[Tuple[int, ...], torch.Tensor, ContextManager, int]] = []
    for stream_index, bucket in enumerate(buckets):
        if use_multiple_cuda_streams:
            stream_context = torch.cuda.StreamContext(state.streams[stream_index])
        with stream_context:
            tensor_bucket: List[torch.Tensor] = [
                (param_list[i].contiguous() if use_contiguous_memory else param_list[i])
                for i in bucket
            ]
            mapping.append(
                (
                    tuple(bucket),
                    tensor_bucket,
                    stream_context,
                    stream_index,
                )
            )

    return Strategy(
        mapping=tuple(mapping),
        reduce_op=reduce_op,
        use_contiguous_memory=use_contiguous_memory,
        src=state.rank if state.is_sender else state.receive_from,
        broadcast=_is_broadcast_necessary(state=state),
    )


def bucket_flatten_(
    model: nn.Module,
    state: DistributedState,
    use_multiple_cuda_streams: bool = False,
    use_contiguous_memory: bool = False,
) -> None:
    """Weights averaging through a bucket-based approach

    All layers are stacked into "buckets" and sent on different CUDA streams in a round-robin fashion

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
    stream_number, stream_index, reduce_op, stream_context = _setup_streams(
        use_multiple_cuda_streams=use_multiple_cuda_streams, state=state
    )

    param_list: List[torch.Tensor] = list(model.parameters())

    buckets: List[List[torch.Tensor]] = [[] for _ in range(stream_number)]
    for layer_index, _ in enumerate(model.parameters()):
        buckets[layer_index % stream_number].append(layer_index)

    for stream_index, bucket in enumerate(buckets):
        if use_multiple_cuda_streams:
            stream_context = torch.cuda.StreamContext(state.streams[stream_index])
        with stream_context:
            layer_list: List[torch.Tensor] = [param_list[i] for i in bucket]
            flatten_bucket: torch.Tensor = _flatten_dense_tensors(layer_list)
            dist.all_reduce(
                tensor=(
                    flatten_bucket.contiguous()
                    if use_contiguous_memory
                    else flatten_bucket
                ),
                op=reduce_op,
                group=state.federated_group[stream_index],
            )
            for layer, updated_layer in zip(
                layer_list,
                _unflatten_dense_tensors(flatten_bucket, layer_list),
            ):
                layer.copy_(updated_layer)
                del updated_layer  # Free memory as soon as possible
            del flatten_bucket  # Free memory as soon as possible


def bucket_coalesced_(
    model: nn.Module,
    state: DistributedState,
    use_multiple_cuda_streams: bool = False,
    use_contiguous_memory: bool = False,
) -> None:
    """Weights averaging through a bucket-based approach

    All layers are stacked into "buckets" and sent on different CUDA streams in a round-robin fashion

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
    stream_number, stream_index, reduce_op, stream_context = _setup_streams(
        use_multiple_cuda_streams=use_multiple_cuda_streams, state=state
    )

    param_list: List[torch.Tensor] = list(model.parameters())

    buckets: List[List[torch.Tensor]] = [[] for _ in range(stream_number)]
    for layer_index, _ in enumerate(model.parameters()):
        buckets[layer_index % stream_number].append(layer_index)

    for stream_index, bucket in enumerate(buckets):
        if use_multiple_cuda_streams:
            stream_context = torch.cuda.StreamContext(state.streams[stream_index])
        with stream_context:
            layer_list: List[torch.Tensor] = [
                (param_list[i].contiguous() if use_contiguous_memory else param_list[i])
                for i in bucket
            ]
            dist.all_reduce_coalesced(
                tensors=layer_list,
                op=reduce_op,
                group=state.federated_group[stream_index],
            )


def bucket_optimized_flatten(
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
    stream_number, stream_index, reduce_op, stream_context = _setup_streams(
        use_multiple_cuda_streams=use_multiple_cuda_streams, state=state
    )

    param_list: List[torch.Tensor] = list(model.parameters())

    bucket_size: int = get_model_size(model=model) // stream_number

    parameter_counter: int = 0
    buckets: List[List[torch.Tensor]] = [[] for _ in range(stream_number)]
    for layer_index, layer in enumerate(model.parameters()):
        buckets[stream_index].append(layer_index)
        parameter_counter += layer.numel()
        if parameter_counter >= bucket_size:
            stream_index += 1
            parameter_counter = 0

    mapping: List[Tuple[Tuple[int, ...], torch.Tensor, ContextManager, int]] = []
    for stream_index, bucket in enumerate(buckets):
        if use_multiple_cuda_streams:
            stream_context = torch.cuda.StreamContext(state.streams[stream_index])
        with stream_context:
            flatten_bucket: torch.Tensor = _flatten_dense_tensors(
                [param_list[i] for i in bucket]
            )
            mapping.append(
                (
                    tuple(bucket),
                    (
                        flatten_bucket.contiguous()
                        if use_contiguous_memory
                        else flatten_bucket
                    ),
                    stream_context,
                    stream_index,
                )
            )

    return Strategy(
        mapping=tuple(mapping),
        reduce_op=reduce_op,
        use_contiguous_memory=use_contiguous_memory,
        src=state.rank if state.is_sender else state.receive_from,
        broadcast=_is_broadcast_necessary(state=state),
    )


def bucket_optimized_coalesced(
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
    stream_number, stream_index, reduce_op, stream_context = _setup_streams(
        use_multiple_cuda_streams=use_multiple_cuda_streams, state=state
    )

    param_list: List[torch.Tensor] = list(model.parameters())

    bucket_size: int = get_model_size(model=model) // stream_number

    parameter_counter: int = 0
    buckets: List[List[torch.Tensor]] = [[] for _ in range(stream_number)]
    for layer_index, layer in enumerate(model.parameters()):
        buckets[stream_index].append(layer_index)
        parameter_counter += layer.numel()
        if parameter_counter >= bucket_size:
            stream_index += 1
            parameter_counter = 0

    mapping: List[Tuple[Tuple[int, ...], torch.Tensor, ContextManager, int]] = []
    for stream_index, bucket in enumerate(buckets):
        if use_multiple_cuda_streams:
            stream_context = torch.cuda.StreamContext(state.streams[stream_index])
        with stream_context:
            tensor_bucket: List[torch.Tensor] = [
                (param_list[i].contiguous() if use_contiguous_memory else param_list[i])
                for i in bucket
            ]
            mapping.append(
                (
                    tuple(bucket),
                    tensor_bucket,
                    stream_context,
                    stream_index,
                )
            )

    return Strategy(
        mapping=tuple(mapping),
        reduce_op=reduce_op,
        use_contiguous_memory=use_contiguous_memory,
        src=state.rank if state.is_sender else state.receive_from,
        broadcast=_is_broadcast_necessary(state=state),
    )


def bucket_optimized_flatten_(
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
    stream_number, stream_index, reduce_op, stream_context = _setup_streams(
        use_multiple_cuda_streams=use_multiple_cuda_streams, state=state
    )

    param_list: List[torch.Tensor] = list(model.parameters())

    bucket_size: int = get_model_size(model=model) // stream_number

    parameter_counter: int = 0
    buckets: List[List[torch.Tensor]] = [[] for _ in range(stream_number)]
    for layer_index, layer in enumerate(model.parameters()):
        buckets[stream_index].append(layer_index)
        parameter_counter += layer.numel()
        if parameter_counter >= bucket_size:
            stream_index += 1
            parameter_counter = 0

    for stream_index, bucket in enumerate(buckets):
        if use_multiple_cuda_streams:
            stream_context = torch.cuda.StreamContext(state.streams[stream_index])
        with stream_context:
            layer_list: List[torch.Tensor] = [param_list[i] for i in bucket]
            flatten_bucket: torch.Tensor = _flatten_dense_tensors(layer_list)
            dist.all_reduce(
                tensor=(
                    flatten_bucket.contiguous()
                    if use_contiguous_memory
                    else flatten_bucket
                ),
                op=reduce_op,
                group=state.federated_group[stream_index],
            )
            for layer, updated_layer in zip(
                layer_list,
                _unflatten_dense_tensors(flatten_bucket, layer_list),
            ):
                layer.copy_(updated_layer)
                del updated_layer  # Free memory as soon as possible
            del flatten_bucket  # Free memory as soon as possible


def bucket_optimized_coalesced_(
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
    stream_number, stream_index, reduce_op, stream_context = _setup_streams(
        use_multiple_cuda_streams=use_multiple_cuda_streams, state=state
    )

    param_list: List[torch.Tensor] = list(model.parameters())

    bucket_size: int = get_model_size(model=model) // stream_number

    parameter_counter: int = 0
    buckets: List[List[torch.Tensor]] = [[] for _ in range(stream_number)]
    for layer_index, layer in enumerate(model.parameters()):
        buckets[stream_index].append(layer_index)
        parameter_counter += layer.numel()
        if parameter_counter >= bucket_size:
            stream_index += 1
            parameter_counter = 0

    for stream_index, bucket in enumerate(buckets):
        if use_multiple_cuda_streams:
            stream_context = torch.cuda.StreamContext(state.streams[stream_index])
        with stream_context:
            layer_list: List[torch.Tensor] = [
                (param_list[i].contiguous() if use_contiguous_memory else param_list[i])
                for i in bucket
            ]
            dist.all_reduce_coalesced(
                tensors=layer_list,
                op=reduce_op,
                group=state.federated_group[stream_index],
            )


def benchmark_aggregation(
    state: DistributedState,
    model: nn.Module,
    iterations: int = 10,
    dump: Optional[str] = None,
) -> None:
    """Benchmark method for testing the available aggregation strategies

    :param state: xFFL distributed state
    :type state: DistributedState
    :param model: PyTorch model
    :type model: nn.Module
    :param iterations: Number of iterations to run each aggreagtion strategy, defaults to 10
    :type iterations: int, optional
    """
    import csv
    import itertools
    import os
    import time

    aggregation_strategies: Tuple[Tuple[callable, callable], ...] = (
        (layer_by_layer, layer_by_layer_),
        (layer_by_layer_optimized, layer_by_layer_optimized_),
        # (bucket_flatten, bucket_flatten_),
        (bucket_coalesced, bucket_coalesced_),
        # (bucket_optimized_flatten, bucket_optimized_flatten_),
        (bucket_optimized_coalesced, bucket_optimized_coalesced_),
    )

    results: List[Tuple[str, str, str, str, str, float, float, float, float]] = []

    for (
        aggregation_strategy_throughput,
        aggregation_strategy_time,
    ) in aggregation_strategies:

        for nccl_algo in [
            "ring",
            "tree",
            "collnet",
            "collnetchain",
            "collnetdirect",
            "nvls",
            "nvlstree",
            "pat",
        ]:

            os.environ["NCCL_ALGO"] = f"{nccl_algo}"

            for nccl_proto in [
                "SIMPLE",
                "LL",
                "LL128",
            ]:

                os.environ["NCCL_PROTO"] = f"{nccl_proto}"
                if state.rank == 0:
                    print("\n")

                for (
                    use_multiple_cuda_streams,
                    use_contiguous_memory,
                ) in itertools.product([False, True], repeat=2):

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # Fundamental to avoid memory fragmentation (really reduces memory consumption here)

                    # Maximum theoretical throughput measurement #
                    strategy: Strategy = aggregation_strategy_throughput(
                        model=model,
                        state=state,
                        use_multiple_cuda_streams=use_multiple_cuda_streams,
                        use_contiguous_memory=use_contiguous_memory,
                    )

                    # Warmup
                    _all_reduce(strategy=strategy, state=state)

                    # Measurement
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start_time = time.perf_counter()
                    for _ in range(iterations):
                        _all_reduce(strategy=strategy, state=state)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    comm_time = (time.perf_counter() - start_time) / iterations

                    # Real aggregation time measurement #
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.empty_cache()  # Fundamental to avoid memory fragmentation (really reduces memory consumption here)

                    # Warmup
                    aggregation_strategy_time(
                        model=model,
                        state=state,
                        use_multiple_cuda_streams=use_multiple_cuda_streams,
                        use_contiguous_memory=use_contiguous_memory,
                    )

                    # Measurement
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start_time = time.perf_counter()
                    for _ in range(iterations):
                        aggregation_strategy_time(
                            model=model,
                            state=state,
                            use_multiple_cuda_streams=use_multiple_cuda_streams,
                            use_contiguous_memory=use_contiguous_memory,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    agg_time = (time.perf_counter() - start_time) / iterations

                    if state.rank == 0:
                        theoretical_throughput: float = (
                            (get_model_size_in_bits(model=model) / comm_time)
                            * (
                                2
                                * (state.federated_world_size - 1)
                                / state.federated_world_size
                            )
                            / 10**9
                        )  # Based on https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md

                        real_throughput: float = (
                            (get_model_size_in_bits(model=model) / agg_time)
                            * (
                                2
                                * (state.federated_world_size - 1)
                                / state.federated_world_size
                            )
                            / 10**9
                        )  # Based on https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md

                        logger.info(
                            f"{aggregation_strategy_throughput.__name__} - NCCL_ALGO={nccl_algo} - NCCL_PROTO={nccl_proto} - Multiple CUDA streams {use_multiple_cuda_streams}, Contiguous memory {use_contiguous_memory}:\n Average communication time over {iterations} iterations: {agg_time:.2f} (max/real adjusted throughput: {theoretical_throughput:.2f}/{real_throughput:.2f} Gb/s - Max GPU RAM allocated: {torch.cuda.max_memory_allocated() / 10**9:.2f} GB)"
                        )

                        results.append(
                            (
                                f"{aggregation_strategy_throughput.__name__}",
                                f"{nccl_algo}",
                                f"{nccl_proto}",
                                f"{use_multiple_cuda_streams}",
                                f"{use_contiguous_memory}",
                                agg_time,
                                theoretical_throughput,
                                real_throughput,
                                torch.cuda.max_memory_allocated() / 10**9,
                            )
                        )

    if state.rank == 0:
        if dump is not None:
            logger.info(f"Dumping benchmarking results to {dump}")
            with open(dump, "w") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Aggregation Strategy",
                        "NCCL Algo",
                        "NCLL Proto",
                        "Multiple CUDA Streams",
                        "Contiguous Memory",
                        "Average Time (s)",
                        "Theoretical Throughput (Gb/s)",
                        "Real Throughput (Gb/s)",
                        "Max GPU RAM Allocated (GB)",
                    ]
                )
                writer.writerows(results)
