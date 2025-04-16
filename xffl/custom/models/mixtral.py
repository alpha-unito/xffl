"""Mixtral modelling"""

import functools
from dataclasses import dataclass
from typing import Callable, Type

from torch.distributed.fsdp import wrap
from transformers import MixtralForCausalLM
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer

from xffl.custom.types import FolderLike

from .model_info import ModelInfo


@dataclass
class Mixtral(ModelInfo):
    """Mixtral essential information"""

    model_type: Type = MixtralForCausalLM
    decoder_layer: Type = MixtralDecoderLayer
    wrapping_policy: Callable = functools.partial(
        wrap.transformer_auto_wrap_policy,
        transformer_layer_cls={decoder_layer},
    )
    path: FolderLike = "/path/to/mixtral/model"
