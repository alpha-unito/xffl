""" LLaMA modelling """

import functools
from dataclasses import dataclass
from typing import Callable, Type

from torch.distributed.fsdp import wrap
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from xffl.custom.types import FolderLike


@dataclass
class llama:
    """LLaMA essential information"""

    model_type: Type[LlamaForCausalLM] = LlamaForCausalLM
    decoder_layer: Type[LlamaDecoderLayer] = LlamaDecoderLayer
    wrapping_policy: Callable = functools.partial(
        wrap.transformer_auto_wrap_policy,
        transformer_layer_cls={decoder_layer},
    )
    path: FolderLike = "/path/to/llama/model"
