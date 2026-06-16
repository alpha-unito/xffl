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
from typing import Any, Mapping, Protocol, TypeAlias

from setup import (
    AnemoiNativeDataset,
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


# ---------------------------------------------------------------------------
# Context creation
# ---------------------------------------------------------------------------


def get_context(
    pretraining_yaml: Path | tuple[Path, ...],
    model_yaml: Path,
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
            )
        )

    return tuple(contexts)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def get_dataset_splits(
    ctx: AnemoiContext,
) -> Mapping[str, AnemoiNativeDataset]:
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

    rollout = ctx.config.training.rollout

    return {
        split_name: AnemoiNativeDataset(
            ctx.dataset,
            multistep=ctx.multistep,
            rollout=rollout,
            start_idx=start_idx,
            end_idx=end_idx,
        )
        for split_name, (start_idx, end_idx) in {
            "train": splits["train"],
            "val": splits["val"],
        }.items()
    }


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def get_aifs_model(
    ctx: AnemoiContext,
    state: DistributedState,
) -> nn.Module:
    """
    Construct an AIFS model for the current rank.

    :param ctx: Rank-specific Anemoi context.
    :param state: Distributed training state.
    :returns: Initialized model.
    :rtype: torch.nn.Module
    """

    return build_model(
        ctx.config,
        ctx.graph_data,
        ctx.statistics,
        ctx.data_indices,
        state.current_device,
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
    targets = batch[:, ctx.multistep, ..., output_idx]

    return inputs, targets
