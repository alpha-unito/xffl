"""Aggregation strategies for local xFFL"""

from logging import Logger, getLogger
from typing import List, Tuple

import torch
import torch.distributed as dist
from torch import nn

from xffl.distributed.distributed_state import DistributedState
from xffl.utils.utils import get_timeout

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def sync_federated_averaging(model: nn.Module, state: DistributedState) -> None:
    """Federated averaging of corresponding model's shards on different hosts

    :param model: PyTorch model
    :type model: nn.Module
    :param state: Partially instantiated distributed state (rank, world_size, backend)
    :type state: DistributedState
    """

    # TODO: contiguous?
    param_list: List[nn.Parameter] = list(model.parameters())  # TODO: maybe module?
    buffer: Tuple[nn.Parameter, torch.Tensor] = (
        param_list[0],  # Tensor 0 has different dimensions
        torch.stack(param_list[1:]),
    )

    reduce_op = dist.ReduceOp.AVG if state.backend == "nccl" else dist.ReduceOp.SUM

    # TODO: Streams (and also process groups) are fixed
    if state.is_hsdp_setup():
        if state.is_sender:
            logger.debug(
                f"[RANK {state.rank}]: All-reduce on {dist.get_process_group_ranks(state.federated_group[0])} + broadcast on {dist.get_process_group_ranks(state.replica_group[0])} with source {state.rank}"
            )
            for param in buffer:
                dist.all_reduce(
                    tensor=param,
                    op=reduce_op,
                    group=state.federated_group[0],
                )
                if not state.backend == "nccl":
                    torch.Tensor.div_(param, state.federated_world_size)
                dist.broadcast(
                    tensor=param,
                    src=state.rank,
                    group=state.replica_group[0],
                )
        else:
            logger.debug(
                f"[RANK {state.rank}]: Broadcast on {dist.get_process_group_ranks(state.replica_group[0])} with source {state.receive_from}"
            )
            for param in buffer:
                dist.broadcast(
                    tensor=param,
                    src=state.receive_from,
                    group=state.replica_group[0],
                )
    else:
        for param in buffer:
            dist.all_reduce(
                tensor=param,
                op=reduce_op,
                group=state.federated_group[0],
            )
            if not state.backend == "nccl":
                torch.Tensor.div_(param, state.federated_world_size)

    for param, updated_param in zip(param_list[1:], buffer[1]):
        param.copy_(updated_param, non_blocking=True)


def sync_federated_averaging_v1(model: nn.Module, state: DistributedState) -> None:
    """Federated averaging of corresponding model's shards on different hosts

    :param model: PyTorch model
    :type model: nn.Module
    :param state: Partially instantiated distributed state (rank, world_size, backend)
    :type state: DistributedState
    """
    replica_world_size: int = (
        min(state.replica_world_size)
        if not isinstance(state.replica_world_size, int)
        else state.replica_world_size
    )
    communicating_processes: int = (
        (state.replica_local_size // replica_world_size)
        if replica_world_size <= state.replica_local_size
        else 1
    )

    bucket = 16

    all_reduce_stream = torch.cuda.default_stream()
    broadcast_stream = torch.cuda.Stream()

    param_list = list(model.parameters())
    buffer = [param_list[0]] + [
        torch.stack(param_list[i : i + bucket])
        for i in range(1, len(param_list), bucket)
    ]

    if (
        communicating_processes * state.replica_rank
        <= state.replica_local_rank
        < communicating_processes * (state.replica_rank + 1)
    ):
        for param in buffer:
            with torch.cuda.StreamContext(all_reduce_stream):
                dist.all_reduce(
                    tensor=param,
                    op=dist.ReduceOp.AVG,
                    group=state.federated_group,
                )
            with torch.cuda.StreamContext(broadcast_stream):
                broadcast_stream.wait_stream(all_reduce_stream)
                dist.broadcast(
                    tensor=param,
                    src=state.rank,
                    group=state.replica_group,
                )
    else:
        federated_local_size: int = (
            state.federated_local_size[state.federated_rank]
            if not isinstance(state.federated_local_size, int)
            else state.federated_local_size
        )
        src: int = (
            state.replica_local_rank + (federated_local_size * state.federated_rank)
            if state.replica_local_rank < communicating_processes * state.replica_rank
            else state.replica_local_rank
            + (state.replica_local_size * (state.replica_rank + 1))
            + (federated_local_size * state.federated_rank)
        )

        for param in buffer:
            dist.broadcast(
                tensor=param,
                src=src,
                group=state.replica_group,
            )

    for index, param in enumerate(param_list[1:], start=0):
        param.copy_(buffer[(index // bucket) + 1][index % bucket])


def sync_federated_averaging_v2(model: nn.Module, state: DistributedState) -> None:
    """Federated averaging of corresponding model's shards on different hosts

    :param model: PyTorch model
    :type model: nn.Module
    :param state: Partially instantiated distributed state (rank, world_size, backend)
    :type state: DistributedState
    """
    replica_world_size: int = (
        min(state.replica_world_size)
        if not isinstance(state.replica_world_size, int)
        else state.replica_world_size
    )
    communicating_processes: int = (
        (state.replica_local_size // replica_world_size)
        if replica_world_size <= state.replica_local_size
        else 1
    )

    # param_list = list(model.parameters())
    # buffer = [layer.contiguous() for layer in param_list]
    all_reduce_stream = torch.cuda.default_stream()
    broadcast_stream = torch.cuda.Stream()

    if (
        communicating_processes * state.replica_rank
        <= state.replica_local_rank
        < communicating_processes * (state.replica_rank + 1)
    ):
        for param in model.parameters():
            with torch.cuda.StreamContext(all_reduce_stream):
                dist.all_reduce(
                    tensor=param,
                    op=dist.ReduceOp.AVG,
                    group=state.federated_group,
                )
            with torch.cuda.StreamContext(broadcast_stream):
                broadcast_stream.wait_stream(all_reduce_stream)
                dist.broadcast(
                    tensor=param,
                    src=state.rank,
                    group=state.replica_group,
                )
    else:
        federated_local_size: int = (
            state.federated_local_size[state.federated_rank]
            if not isinstance(state.federated_local_size, int)
            else state.federated_local_size
        )
        src: int = (
            state.replica_local_rank + (federated_local_size * state.federated_rank)
            if state.replica_local_rank < communicating_processes * state.replica_rank
            else state.replica_local_rank
            + (state.replica_local_size * (state.replica_rank + 1))
            + (federated_local_size * state.federated_rank)
        )

        for param in model.parameters():
            dist.broadcast(
                tensor=param,
                src=src,
                group=state.replica_group,
            )


def layer_by_layer_aggregation(model: nn.Module, state: DistributedState) -> None:
    """Federated averaging of corresponding model's shards on different hosts

    :param model: PyTorch model
    :type model: nn.Module
    :param state: Partially instantiated distributed state (rank, world_size, backend)
    :type state: DistributedState
    """
    if state.is_hsdp_setup():
        if state.is_sender:
            ag_group = dist.get_process_group_ranks(state.federated_group[0])
            bc_group = dist.get_process_group_ranks(state.replica_group[0])
            logger.warning(
                f"[RANK {state.rank}] AllGather to {ag_group} and Broadcast to {bc_group} with source {state.rank}"
            )

            for index, param in enumerate(model.parameters()):
                if state.backend == "nccl":
                    stream_index = index % len(state.streams)
                    with torch.cuda.StreamContext(state.streams[stream_index]):
                        dist.all_reduce(
                            tensor=param,
                            op=dist.ReduceOp.AVG,
                            group=state.federated_group[stream_index],
                        )
                        if (
                            len(dist.get_process_group_ranks(state.replica_group[0]))
                            > 1
                        ):
                            dist.broadcast(
                                tensor=param,
                                src=state.rank,
                                group=state.replica_group[stream_index],
                            )
                else:
                    dist.all_reduce(
                        tensor=param,
                        op=dist.ReduceOp.SUM,
                        group=state.federated_group[0],
                    )
                    torch.Tensor.div_(param, state.federated_world_size)
                    if len(dist.get_process_group_ranks(state.replica_group[0])) > 1:
                        dist.broadcast(
                            tensor=param,
                            src=state.rank,
                            group=state.replica_group[0],
                        )
            logger.warning(f"[RANK {state.rank}] FATTO!")
        else:
            bc_group = dist.get_process_group_ranks(state.replica_group[0])
            logger.warning(
                f"[RANK {state.rank}] Broadcast to {bc_group} from {state.receive_from}"
            )

            if len(dist.get_process_group_ranks(state.replica_group[0])) > 1:
                for index, param in enumerate(model.parameters()):
                    if state.backend == "nccl":
                        stream_index = index % len(state.streams)
                        with torch.cuda.StreamContext(state.streams[stream_index]):
                            dist.broadcast(
                                tensor=param,
                                src=state.receive_from,
                                group=state.replica_group[stream_index],
                            )
                    else:
                        dist.broadcast(
                            tensor=param,
                            src=state.receive_from,
                            group=state.replica_group[0],
                        )
                logger.warning(f"[RANK {state.rank}] FATTO!")
    else:  # FSDP
        for index, param in enumerate(model.parameters()):
            if state.backend == "nccl":
                stream_index = index % len(state.streams)
                with torch.cuda.StreamContext(state.streams[stream_index]):
                    dist.all_reduce(
                        tensor=param,
                        op=dist.ReduceOp.AVG,
                        group=state.federated_group[stream_index],
                    )
            else:
                dist.all_reduce(
                    tensor=param,
                    op=dist.ReduceOp.SUM,
                    group=state.federated_group[0],
                )
                torch.Tensor.div_(param, state.federated_world_size)


def sync_federated_averaging_v4(model: nn.Module, state: DistributedState) -> None:
    """Federated averaging of corresponding model's shards on different hosts

    :param model: PyTorch model
    :type model: nn.Module
    :param state: Partially instantiated distributed state (rank, world_size, backend)
    :type state: DistributedState
    """
    replica_world_size: int = (
        min(state.replica_world_size)
        if not isinstance(state.replica_world_size, int)
        else state.replica_world_size
    )
    communicating_processes: int = (
        (state.replica_local_size // replica_world_size)
        if replica_world_size <= state.replica_local_size
        else 1
    )

    bucket = 16

    param_list = list(model.parameters())
    bucket_len = (len(param_list) - 1) // bucket

    if (
        communicating_processes * state.replica_rank
        <= state.replica_local_rank
        < communicating_processes * (state.replica_rank + 1)
    ):
        for index in range(bucket + 1):
            with torch.cuda.StreamContext(state.streams[index % len(state.streams)]):
                if index == 0:
                    param = param_list[0]
                else:
                    begin = ((index - 1) * bucket_len) + 1
                    end = (index * bucket_len) + 1
                    param = torch.stack(param_list[begin:end])

                dist.all_reduce(
                    tensor=param,
                    op=dist.ReduceOp.AVG,
                    group=state.federated_group[index % len(state.federated_group)],
                )
                dist.broadcast(
                    tensor=param,
                    src=state.rank,
                    group=state.replica_group[index % len(state.replica_group)],
                )
                if index > 0:
                    begin = ((index - 1) * bucket_len) + 1
                    end = (index * bucket_len) + 1
                    for i, layer in enumerate(param_list[begin:end], start=0):
                        layer.copy_(param[i])

    else:
        federated_local_size: int = (
            state.federated_local_size[state.federated_rank]
            if not isinstance(state.federated_local_size, int)
            else state.federated_local_size
        )
        src: int = (
            state.replica_local_rank + (federated_local_size * state.federated_rank)
            if state.replica_local_rank < communicating_processes * state.replica_rank
            else state.replica_local_rank
            + (state.replica_local_size * (state.replica_rank + 1))
            + (federated_local_size * state.federated_rank)
        )

        for index in range(bucket + 1):
            with torch.cuda.StreamContext(state.streams[index % len(state.streams)]):
                if index == 0:
                    param = param_list[0]
                else:
                    begin = ((index - 1) * bucket_len) + 1
                    end = (index * bucket_len) + 1
                    param = torch.stack(param_list[begin:end])

                dist.broadcast(
                    tensor=param,
                    src=src,
                    group=state.replica_group[index % len(state.replica_group)],
                )
                if index > 0:
                    begin = ((index - 1) * bucket_len) + 1
                    end = (index * bucket_len) + 1
                    for i, layer in enumerate(param_list[begin:end], start=0):
                        layer.copy_(param[i])


def async_federated_averaging(
    model: nn.Module, state: DistributedState
) -> Tuple[List[dist.Work], List[torch.Tensor]]:
    """Federated averaging of corresponding model's shards on different hosts

    :param model: PyTorch model
    :type model: nn.Module
    :param state: Partially instantiated distributed state (rank, world_size, backend)
    :type state: DistributedState
    """
    replica_world_size: int = (
        min(state.replica_world_size)
        if not isinstance(state.replica_world_size, int)
        else state.replica_world_size
    )
    communicating_processes: int = (
        (state.replica_local_size // replica_world_size)
        if replica_world_size <= state.replica_local_size
        else 1
    )

    param_list = list(model.parameters())
    buffer = [
        param_list[0],  # Tensor 0 has different dimensions
        torch.stack(param_list[1:]),
    ]

    # pre_values = []
    # post_values = []
    # differences = []

    # for p in model.parameters():
    #    if p.requires_grad:  # only change learnable parameters
    #        with torch.no_grad():
    #            p.fill_(state.federated_rank)

    # for param in model.parameters():
    #    pre_values.append(param.clone().detach().cpu())

    all_reduce_handles: List[dist.Work] = []
    broadcast_handles: List[dist.Work] = []
    if (
        communicating_processes * state.replica_rank
        <= state.replica_local_rank
        < communicating_processes * (state.replica_rank + 1)
    ):
        for param in buffer:
            all_reduce_handles.append(
                dist.all_reduce(
                    tensor=param.contiguous(),
                    op=dist.ReduceOp.AVG,
                    group=state.federated_group,
                    async_op=True,
                )
            )
        for all_reduce_handle, param in zip(all_reduce_handles, buffer):
            all_reduce_handle.wait(timeout=get_timeout(seconds=60))
            broadcast_handles.append(
                dist.broadcast(
                    tensor=param.contiguous(),
                    src=state.rank,
                    group=state.replica_group,
                    async_op=True,
                )
            )
    else:
        federated_local_size: int = (
            state.federated_local_size[state.federated_rank]
            if not isinstance(state.federated_local_size, int)
            else state.federated_local_size
        )
        src: int = (
            state.replica_local_rank + (federated_local_size * state.federated_rank)
            if state.replica_local_rank < communicating_processes * state.replica_rank
            else state.replica_local_rank
            + (state.replica_local_size * (state.replica_rank + 1))
            + (federated_local_size * state.federated_rank)
        )

        for param in buffer:
            broadcast_handles.append(
                dist.broadcast(
                    tensor=param.contiguous(),
                    src=src,
                    group=state.replica_group,
                    async_op=True,
                )
            )

    return broadcast_handles, buffer

    # for param in model.parameters():
    #    post_values.append(param.clone().detach().cpu())

    # for layer, _ in enumerate(model.parameters()):
    #    differences.append(pre_values[layer] - post_values[layer])

    # logger.warning(f"[RANK {state.rank}]:\n"
    #               f"pre-values: {pre_values}\n"
    #               f"post-values: {post_values}\n"
    #               f"differences: {differences}\n")
