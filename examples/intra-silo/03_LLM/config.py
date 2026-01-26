"""Configuration file for the xFFL-LLM example"""

import functools
import logging
from dataclasses import dataclass
from typing import Tuple

from torch.distributed.fsdp import wrap
from transformers import LlamaForCausalLM, MixtralForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer

from xffl.custom.config import DatasetInfo, ModelInfo
from xffl.custom.types import FileLike, FolderLike, PathLike

TINY_RANDOM_LLAMA_3: str = "tiny-random-llama-3"
LLAMA3_1_8B: str = "llama3.1-8b"
LLAMA3_1_70B: str = "llama3.1-70b"
MIXTRAL_8x7b_v0_1: str = "mixtral-8x7b-v0.1"


CLEAN_MC4_IT: str = "clean_mc4_it"


@dataclass
class llama(ModelInfo):
    name = LLAMA3_1_8B
    model_type = LlamaForCausalLM
    decoder_layers = (LlamaDecoderLayer,)
    wrapping_policy = (
        functools.partial(
            wrap.transformer_auto_wrap_policy,
            transformer_layer_cls={decoder_layers},
        ),
    )
    path = "/leonardo_scratch/fast/uToID_bench/xffl/models/"


@dataclass
class mixtral(ModelInfo):
    name = MIXTRAL_8x7b_v0_1
    model_type = MixtralForCausalLM
    decoder_layers = MixtralDecoderLayer
    wrapping_policy = functools.partial(
        wrap.transformer_auto_wrap_policy,
        transformer_layer_cls={decoder_layers},
    )
    path = "/leonardo_scratch/fast/uToID_bench/xffl/models/"


@dataclass
class cleanmc4it(DatasetInfo):
    name = CLEAN_MC4_IT
    splits = {
        "train": "train",
        "val": "val",
    }
    path = "/leonardo_scratch/fast/uToID_bench/xffl/data/"


@dataclass
class xffl_config:
    model: ModelInfo = llama
    dataset: DatasetInfo = cleanmc4it
    output: FolderLike = None
    loglevel: int = logging.DEBUG
    seed: int = 42
    hsdp: int = None
    federated_scaling: int = None
    cuda_streams: int = 1
    wandb: bool = False
    wandb_name: str = "LLaMA-3.1 8B"
    wandb_mode: str = "disabled"
    online: bool = False
    attention: str = "sdpa"
    subsampling: int = 32
    train_batch_size: int = 2
    val_batch_size: int = 1
    workers: int = 2
    scale_learning_rate: bool = False
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: Tuple[float] = (0.9, 0.95)
    benchmark: int = 0
    workspace: FolderLike = None
    csv: FileLike = None
    output: PathLike = "/output/"
    output_model: str = "output"
    epochs: int = 1
    federated_batches: int = 8
