"""LLaMA modelling"""

import functools
from dataclasses import dataclass
from typing import Callable, Type

import transformers.models.llama.modeling_llama
from torch.distributed.fsdp import wrap
from transformers import LlamaForCausalLM

from xffl.custom.types import FolderLike

from .model_info import ModelInfo


@dataclass
class Llama(ModelInfo):
    """LLaMA essential information"""

    model_type: Type[LlamaForCausalLM] = LlamaForCausalLM
    decoder_layer: Type[transformers.models.llama.modeling_llama.LlamaDecoderLayer] = (
        transformers.models.llama.modeling_llama.LlamaDecoderLayer
    )
    wrapping_policy: Callable = functools.partial(
        wrap.transformer_auto_wrap_policy,
        transformer_layer_cls={decoder_layer},
    )
    path: FolderLike = "/path/to/llama/model"
