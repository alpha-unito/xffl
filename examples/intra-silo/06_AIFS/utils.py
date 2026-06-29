"""
Utility functions for building and managing Anemoi contexts.

This module centralizes all logic related to:

* loading Anemoi configurations;
* loading datasets and graph structures;
* constructing model instances;
* creating dataset splits;
* computing loss weights;
* preprocessing training batches.

Each federated rank is associated with exactly one
:class:`AnemoiContext` instance.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Tuple, TypeAlias

import torch
from setup import (
    AnemoiNativeDataset,
    advance_input,
    build_indices,
    build_loss_weights,
    build_model,
    load_config,
    load_data,
    load_graph,
    split_dataset,
)
from torch import Tensor, nn

from xffl.distributed.distributed_state import DistributedState

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ConfigType: TypeAlias = Any
OmegaConfType: TypeAlias = Any
DatasetType: TypeAlias = Any
StatisticsType: TypeAlias = Any
GraphDataType: TypeAlias = Any
DataIndicesType: TypeAlias = Any


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class HasPreProcessors(Protocol):
    """Protocol for models exposing Anemoi pre-processing."""

    def pre_processors(
        self,
        batch: Tensor,
        in_place: bool = False,
    ) -> Tensor: ...


# ---------------------------------------------------------------------------
# Context container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AnemoiContext:
    """
    Container holding all resources required by a federated silo.

    Each federated rank is associated with exactly one
    :class:`AnemoiContext` instance.

    :ivar config: Runtime Anemoi configuration.
    :ivar config_omegaconf: Original OmegaConf configuration.
    :ivar dataset: Loaded dataset.
    :ivar name_to_index: Mapping from variable names to indices.
    :ivar statistics: Dataset normalization statistics.
    :ivar graph_data: Graph structure used by the model.
    :ivar data_indices: Input/output feature indices.
    :ivar multistep: Number of input timesteps used by the model.
    """

    config: ConfigType
    config_omegaconf: OmegaConfType
    dataset: DatasetType
    name_to_index: Mapping[str, int]
    statistics: StatisticsType
    graph_data: GraphDataType
    data_indices: DataIndicesType
    multistep: int
    warmup_steps: Optional[int] = None
    boundary_mask: Optional[bool] = None
    current_rollout: Optional[int] = None
    rollout_start: Optional[int] = None
    rollout_increment: Optional[int] = None
    rollout_max: Optional[int] = None
    finetuning: bool = False


# ---------------------------------------------------------------------------
# Context creation
# ---------------------------------------------------------------------------


def get_context(
    pretraining_yaml: Path | tuple[Path, ...],
    model_yaml: Path,
    finetuning: bool = False,
) -> tuple[AnemoiContext, ...]:
    """
    Build one :class:`AnemoiContext` for each provided pretraining
    configuration.

    :param pretraining_yaml: Single pretraining YAML path or tuple of paths. Each path corresponds to one federated silo.
    :param model_yaml: Path to the model configuration file.
    :returns: Tuple containing one context for each provided YAML file.
    :rtype: tuple[AnemoiContext, ...]
    """

    if isinstance(pretraining_yaml, Path):
        pretraining_yaml = (pretraining_yaml,)

    contexts: list[AnemoiContext] = []

    for yaml_path in pretraining_yaml:
        config, config_omegaconf = load_config(
            yaml_path,
            model_yaml,
        )

        graph_data = load_graph(config)

        dataset, name_to_index, statistics = load_data(config)

        data_indices = build_indices(
            config_omegaconf,
            name_to_index,
        )

        cutout_mask = graph_data[config.graph.data]["cutout_mask"].squeeze().bool()

        contexts.append(
            AnemoiContext(
                config=config,
                config_omegaconf=config_omegaconf,
                dataset=dataset,
                name_to_index=name_to_index,
                statistics=statistics,
                graph_data=graph_data,
                data_indices=data_indices,
                multistep=config.training.multistep_input,
                warmup_steps=config.training.warmup_steps if finetuning else None,
                boundary_mask=~cutout_mask if finetuning else None,
                rollout_start=(config.training.rollout.start if finetuning else None),
                rollout_increment=(
                    config.training.rollout.epoch_increment if finetuning else None
                ),
                rollout_max=config.training.rollout.max if finetuning else None,
                finetuning=finetuning,
            )
        )

    return tuple(contexts)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def get_dataset_splits(
    ctx: AnemoiContext, batch_size: int, rollout: Optional[int] = None
) -> Tuple[Mapping[str, AnemoiNativeDataset], Optional[int]]:
    """
    Create train and validation dataset splits.

    :param ctx: Context associated with the current federated silo.

    :returns: Mapping containing the ``train`` and ``val`` datasets.
    :rtype: Mapping[str, AnemoiNativeDataset]
    """

    splits = split_dataset(
        ctx.dataset,
        val_years=2,
        test_years=1,
    )

    steps_per_epoch_est: Optional[int] = None
    if ctx.finetuning:
        assert ctx.multistep is not None
        assert ctx.rollout_start is not None

        initial_samples = (
            (splits["train"][1] - splits["train"][0])
            - ctx.multistep
            - ctx.rollout_start
            + 1
        )
        steps_per_epoch_est = (initial_samples + batch_size - 1) // batch_size

    if not ctx.finetuning:
        _rollout = ctx.config.training.rollout
    else:
        if rollout is None:
            _rollout = ctx.config.training.rollout.start
        else:
            _rollout = rollout

    return {
        split_name: AnemoiNativeDataset(
            ctx.dataset,
            multistep=ctx.multistep,
            rollout=_rollout,
            start_idx=start_idx,
            end_idx=end_idx,
        )
        for split_name, (start_idx, end_idx) in {
            "train": splits["train"],
            "val": splits["val"],
        }.items()
    }, steps_per_epoch_est


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def get_aifs_model(
    ctx: AnemoiContext, state: DistributedState, checkpoint_path: Optional[Path] = None
) -> nn.Module:
    """
    Construct an AIFS model for the current rank.

    :param ctx: Rank-specific Anemoi context.
    :param state: Distributed training state.
    :returns: Initialized model.
    :rtype: torch.nn.Module
    """

    return build_model(
        config=ctx.config,
        graph_data=ctx.graph_data,
        statistics=ctx.statistics,
        data_indices=ctx.data_indices,
        device=state.current_device,
        checkpoint_path=checkpoint_path,
    )


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def get_var_node_weights(
    ctx: AnemoiContext,
    state: DistributedState,
) -> tuple[Tensor, Tensor]:
    """
    Compute variable and node weights used by the weighted MSE loss.

    :param ctx: Rank-specific Anemoi context.
    :param state: Distributed training state.
    :returns: Tuple containing variable weights and node weights.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """

    var_weights, node_weights = build_loss_weights(
        ctx.config,
        ctx.data_indices,
        ctx.graph_data,
        state.current_device,
    )

    cutout_mask = (
        ctx.graph_data[ctx.config.graph.data]["cutout_mask"]
        .squeeze()
        .bool()
        .to(state.current_device)
    )

    node_weights *= cutout_mask.float()

    return var_weights, node_weights


# ---------------------------------------------------------------------------
# Batch preprocessing
# ---------------------------------------------------------------------------


def pre_process_hook(
    model: HasPreProcessors,
    batch: Tensor,
    state: DistributedState,
    ctx: AnemoiContext,
    rstep: int = 0,
) -> tuple[Tensor, Tensor]:
    """
    Convert a raw batch into model inputs and prediction targets.

    The input batch is first moved to the current device and
    preprocessed through the model-specific preprocessing pipeline.

    :param model: Model exposing the ``pre_processors`` method.
    :param batch: Raw input batch.
    :param state: Distributed training state.
    :param ctx: Rank-specific Anemoi context.
    :returns: Tuple ``(inputs, targets)`` ready for training.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """

    input_idx = ctx.data_indices.internal_data.input.full
    output_idx = ctx.data_indices.internal_data.output.full

    batch = batch.to(
        device=state.current_device,
        non_blocking=True,
    )

    batch = model.pre_processors(
        batch,
        in_place=False,
    )

    inputs = batch[:, : ctx.multistep, ..., input_idx]
    targets = batch[:, ctx.multistep + rstep, ..., output_idx]

    return inputs, targets


def finetuning_pre_process_hook(
    model: HasPreProcessors,
    batch: Tensor,
    state: DistributedState,
    ctx: AnemoiContext,
    rstep: Optional[int] = None,
    rollout: Optional[int] = None,
    output: Optional[Tensor] = None,
    prev_data: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """
    Convert a raw batch into model inputs and prediction targets.

    The input batch is first moved to the current device and
    preprocessed through the model-specific preprocessing pipeline.

    :param model: Model exposing the ``pre_processors`` method.
    :param batch: Raw input batch.
    :param state: Distributed training state.
    :param ctx: Rank-specific Anemoi context.
    :returns: Tuple ``(inputs, targets)`` ready for training.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """

    input_idx = ctx.data_indices.internal_data.input.full
    output_idx = ctx.data_indices.internal_data.output.full

    batch = batch.to(
        device=state.current_device,
        non_blocking=True,
    )

    batch = model.pre_processors(
        batch,
        in_place=False,
    )

    if rstep is None:
        rstep = 0
    if rollout is None:
        rollout = 1

    targets = batch[:, ctx.multistep + rstep, ..., output_idx]

    if rstep == 0:
        inputs = batch[:, : ctx.multistep, ..., input_idx]
    elif 0 < rstep <= rollout - 1:
        inputs = advance_input(
            prev_data, output, batch, rstep - 1, ctx.multistep, ctx.data_indices
        )
        next_ts = ctx.multistep + rstep
        inputs[:, -1, :, ctx.boundary_mask, :] = batch[
            :, next_ts, :, ctx.boundary_mask, :
        ][:, :, :, input_idx].to(device="cuda:0")

    return inputs, targets


# ---------------------------------------------------------------------------
# Custom loss
# ---------------------------------------------------------------------------


def weighted_mse_loss(y_pred, y, var_weights, node_weights):

    err = torch.square(y_pred.float() - y.float())  # (y_pred - y)^2
    err = (
        err * var_weights
    )  # moltiplico ogni errore per il peso della variabile corrispondente
    err = torch.mean(
        err, dim=-1
    )  # media tra tutti gli elementi del tensore per ogni punto della griglia
    err = (
        err * node_weights
    )  # moltiplico ogni errore per il peso del nodo corrispondente
    return torch.sum(err) / torch.sum(node_weights.expand_as(err))


def post_rollout_hook(
    batch: Tensor,
    output: Tensor,
    ctx: AnemoiContext,
    rstep: int,
    rollout: int,
    device: torch.device,
) -> Tensor:
    """

    :param model: Model exposing the ``pre_processors`` method.
    :param batch: Raw input batch.
    :param state: Distributed training state.
    :param ctx: Rank-specific Anemoi context.
    :returns: Tuple ``(inputs, targets)`` ready for training.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """

    input_idx = ctx.data_indices.internal_data.input.full

    x = batch[:, : ctx.multistep, ..., input_idx]

    if rstep < rollout - 1:
        x = advance_input(
            x, output, batch, rstep, ctx.multistep, ctx.data_indices, device
        )
        next_ts = ctx.multistep + rstep + 1
        x[:, -1, :, ctx.boundary_mask, :] = batch[:, next_ts, :, ctx.boundary_mask, :][  # type: ignore
            :, :, :, input_idx
        ]

    return x
