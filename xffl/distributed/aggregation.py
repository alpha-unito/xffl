"""Aggregation strategies for local xFFL"""

from contextlib import nullcontext
from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Callable, ContextManager, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import cuda, nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.distributed import ProcessGroup

from xffl.distributed.distributed_state import DistributedState
from xffl.learning.utils import get_model_size, get_model_size_in_bits

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

    mapping: Tuple[
        Tuple[
            Tuple[int, ...],
            Tuple[torch.Tensor, ...] | torch.Tensor,
            ContextManager,
            int,
        ],
        ...,
    ]
    reduce_op: dist.ReduceOp.RedOpType
    use_contiguous_memory: bool
    src: int
    broadcast: bool


# --------------------------------------------------------------------------- #
#                               Helper functions                              #
# --------------------------------------------------------------------------- #


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
    stream_context: ContextManager = nullcontext  # type: ignore
    if state.backend == "nccl":
        reduce_op = dist.ReduceOp.AVG
        stream_context = torch.cuda.StreamContext(cuda.default_stream())

    return stream_number, stream_index, reduce_op, stream_context


def _get_stream_context(state: DistributedState, stream_index: int) -> ContextManager:
    """Checks if multiple CUDA streams are initialized and returns the appropriate context manager.

    :param state: XFFL distributed state
    :type state: DistributedState
    :param index: CUDA stream index to instantiate
    :type index: int
    :return: CUDE stream context manager
    :rtype: ContextManager
    """
    stream_context: ContextManager = nullcontext  # type: ignore
    if state.streams is not None:
        if stream_index < len(state.streams):
            stream_context = torch.cuda.StreamContext(state.streams[stream_index])
        else:
            logger.error(
                f"{len(state.streams)} CUDA streams are enabled, but stream {stream_index} is requested. Falling back to the default stream."
            )
    else:
        logger.error(
            "Multiple CUDA streams are enabled, but are not available. Falling back to the default stream."
        )

    return stream_context


def _get_src_rank(state: DistributedState) -> int:
    """Get the source rank for the aggregation.

    :param state: XFFL distributed state
    :type state: DistributedState
    :raises AttributeError: If the distributed state configuration of the calling process is incomplete
    :return: The source rank for the aggregation
    :rtype: int
    """
    src_rank: Optional[int] = state.rank if state.is_sender else state.receive_from
    if src_rank is None:
        logger.critical(
            f"Impossible to establish the source rank for Rank {state.rank} with configuration is_sender {state.is_sender} and receive_from {state.receive_from}"
        )
        raise AttributeError(obj=state)
    return src_rank


def _get_federated_group(state: DistributedState, group_index: int) -> ProcessGroup:
    """Get the federated group on which to aggregate.

    :param state: XFFL distributed state
    :type state: DistributedState
    :param group_index: Index of the federated group
    :type group_index: int
    :raises AttributeError: If the distributed state is not correctly initialized
    :return: The federated group for the aggregation
    :rtype: ProcessGroup
    """
    if state.federated_group is not None:
        if group_index < len(state.federated_group):
            return state.federated_group[group_index]
        else:
            logger.error(
                f"{len(state.federated_group)} federated groups are enabled, but group {group_index} is requested."
            )
            raise AttributeError(obj=state)
    else:
        logger.error(
            "Federated scaling is enabled, but the federated groups have not been correctly initialized."
        )
        raise AttributeError(obj=state)


def _is_broadcast_necessary(state: DistributedState) -> bool:
    """Checks if the current rank needs to take part into a weights broadcast

    :param state: xFFL distributed state (rank, world_size, backend)
    :type state: DistributedState
    """
    return (
        state.replica_group is not None
        and len(dist.get_process_group_ranks(state.replica_group[0])) > 1
    )


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
    assert state.federated_group is not None

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


# --------------------------------------------------------------------------- #
#                               Strategy-based                                #
# --------------------------------------------------------------------------- #


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
            stream_context = _get_stream_context(state=state, stream_index=stream_index)
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
        src=_get_src_rank(state),
        broadcast=_is_broadcast_necessary(state=state),
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
            stream_context = _get_stream_context(state=state, stream_index=stream_index)
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
        src=_get_src_rank(state),
        broadcast=_is_broadcast_necessary(state=state),
    )


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

    buckets: List[List[int]] = [[] for _ in range(stream_number)]
    for layer_index, _ in enumerate(model.parameters()):
        buckets[layer_index % stream_number].append(layer_index)

    mapping: List[Tuple[Tuple[int, ...], torch.Tensor, ContextManager, int]] = []
    for stream_index, bucket in enumerate(buckets):
        if use_multiple_cuda_streams:
            stream_context = _get_stream_context(state=state, stream_index=stream_index)
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
        src=_get_src_rank(state),
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

    buckets: List[List[int]] = [[] for _ in range(stream_number)]
    for layer_index, _ in enumerate(model.parameters()):
        buckets[layer_index % stream_number].append(layer_index)

    mapping: List[
        Tuple[Tuple[int, ...], Tuple[torch.Tensor, ...], ContextManager, int]
    ] = []
    for stream_index, bucket in enumerate(buckets):
        if use_multiple_cuda_streams:
            stream_context = _get_stream_context(state=state, stream_index=stream_index)
        with stream_context:
            tensor_bucket: Tuple[torch.Tensor, ...] = tuple(
                [
                    (
                        param_list[i].contiguous()
                        if use_contiguous_memory
                        else param_list[i]
                    )
                    for i in bucket
                ]
            )
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
        src=_get_src_rank(state),
        broadcast=_is_broadcast_necessary(state=state),
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
    buckets: List[List[int]] = [[] for _ in range(stream_number)]
    for layer_index, layer in enumerate(model.parameters()):
        buckets[stream_index].append(layer_index)
        parameter_counter += layer.numel()
        if parameter_counter >= bucket_size:
            stream_index += 1
            parameter_counter = 0

    mapping: List[Tuple[Tuple[int, ...], torch.Tensor, ContextManager, int]] = []
    for stream_index, bucket in enumerate(buckets):
        if use_multiple_cuda_streams:
            stream_context = _get_stream_context(state=state, stream_index=stream_index)
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
        src=_get_src_rank(state),
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
    buckets: List[List[int]] = [[] for _ in range(stream_number)]
    for layer_index, layer in enumerate(model.parameters()):
        buckets[stream_index].append(layer_index)
        parameter_counter += layer.numel()
        if parameter_counter >= bucket_size:
            stream_index += 1
            parameter_counter = 0

    mapping: List[
        Tuple[Tuple[int, ...], Tuple[torch.Tensor, ...], ContextManager, int]
    ] = []
    for stream_index, bucket in enumerate(buckets):
        if use_multiple_cuda_streams:
            stream_context = _get_stream_context(state=state, stream_index=stream_index)
        with stream_context:
            tensor_bucket: Tuple[torch.Tensor, ...] = tuple(
                [
                    (
                        param_list[i].contiguous()
                        if use_contiguous_memory
                        else param_list[i]
                    )
                    for i in bucket
                ]
            )
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
        src=_get_src_rank(state),
        broadcast=_is_broadcast_necessary(state=state),
    )


# --------------------------------------------------------------------------- #
#                               In-place based                                #
# --------------------------------------------------------------------------- #


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
            stream_context = _get_stream_context(state=state, stream_index=stream_index)
        with stream_context:
            dist.all_reduce(
                tensor=param.contiguous() if use_contiguous_memory else param,
                op=reduce_op,
                group=(
                    state.federated_group[stream_index]
                    if state.federated_group is not None
                    else None
                ),
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
            stream_context = _get_stream_context(state=state, stream_index=stream_index)
        with stream_context:
            dist.all_reduce(
                tensor=layer.contiguous() if use_contiguous_memory else layer,
                op=reduce_op,
                group=_get_federated_group(state=state, group_index=stream_index),
            )
            parameter_counter += layer.numel()
            if parameter_counter >= bucket_size:
                stream_index += 1
                parameter_counter = 0


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

    buckets: List[List[int]] = [[] for _ in range(stream_number)]
    for layer_index, _ in enumerate(model.parameters()):
        buckets[layer_index % stream_number].append(layer_index)

    for stream_index, bucket in enumerate(buckets):
        if use_multiple_cuda_streams:
            stream_context = _get_stream_context(state=state, stream_index=stream_index)
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
                group=_get_federated_group(state=state, group_index=stream_index),
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

    buckets: List[List[int]] = [[] for _ in range(stream_number)]
    for layer_index, _ in enumerate(model.parameters()):
        buckets[layer_index % stream_number].append(layer_index)

    for stream_index, bucket in enumerate(buckets):
        if use_multiple_cuda_streams:
            stream_context = _get_stream_context(state=state, stream_index=stream_index)
        with stream_context:
            layer_list: List[torch.Tensor] = [
                (param_list[i].contiguous() if use_contiguous_memory else param_list[i])
                for i in bucket
            ]
            dist.all_reduce_coalesced(
                tensors=layer_list,
                op=reduce_op,
                group=_get_federated_group(state=state, group_index=stream_index),
            )


def bucket_optimized_flatten_(
    model: nn.Module,
    state: DistributedState,
    use_multiple_cuda_streams: bool = False,
    use_contiguous_memory: bool = False,
) -> None:
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
    buckets: List[List[int]] = [[] for _ in range(stream_number)]
    for layer_index, layer in enumerate(model.parameters()):
        buckets[stream_index].append(layer_index)
        parameter_counter += layer.numel()
        if parameter_counter >= bucket_size:
            stream_index += 1
            parameter_counter = 0

    for stream_index, bucket in enumerate(buckets):
        if use_multiple_cuda_streams:
            stream_context = _get_stream_context(state=state, stream_index=stream_index)
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
                group=_get_federated_group(state=state, group_index=stream_index),
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
) -> None:
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
    buckets: List[List[int]] = [[] for _ in range(stream_number)]
    for layer_index, layer in enumerate(model.parameters()):
        buckets[stream_index].append(layer_index)
        parameter_counter += layer.numel()
        if parameter_counter >= bucket_size:
            stream_index += 1
            parameter_counter = 0

    for stream_index, bucket in enumerate(buckets):
        if use_multiple_cuda_streams:
            stream_context = _get_stream_context(state=state, stream_index=stream_index)
        with stream_context:
            layer_list: List[torch.Tensor] = [
                (param_list[i].contiguous() if use_contiguous_memory else param_list[i])
                for i in bucket
            ]
            dist.all_reduce_coalesced(
                tensors=layer_list,
                op=reduce_op,
                group=_get_federated_group(state=state, group_index=stream_index),
            )


# --------------------------------------------------------------------------- #
#                               Benchmark code                                #
# --------------------------------------------------------------------------- #


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

    aggregation_strategies: Tuple[Tuple[Callable, Callable], ...] = (
        (layer_by_layer, layer_by_layer_),
        (layer_by_layer_optimized, layer_by_layer_optimized_),
        (bucket_flatten, bucket_flatten_),
        (bucket_coalesced, bucket_coalesced_),
        (bucket_optimized_flatten, bucket_optimized_flatten_),
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
                                * (state.federated_world_size - 1)  # type: ignore
                                / state.federated_world_size
                            )
                            / 10**9
                        )  # Based on https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md

                        real_throughput: float = (
                            (get_model_size_in_bits(model=model) / agg_time)
                            * (
                                2
                                * (state.federated_world_size - 1)  # type: ignore
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
