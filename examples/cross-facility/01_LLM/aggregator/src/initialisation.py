from typing import Optional

import torch
from transformers import AutoConfig, AutoModel

from xffl.learning import utils

# Sets RNGs seeds and force PyTorch's deterministic execution
generator: Optional[torch.Generator] = utils.set_deterministic_execution(seed=42)

print("Downloading model configuration...")
config = AutoConfig.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    use_cache=True,
    local_files_only=True,
)

print("Creating model...")
model = AutoModel.from_config(config)

print("Forcing weights initialization...")
model.apply(model._init_weights)

print("Saving...")
model.save_pretrained(
    save_directory="/leonardo_scratch/fast/uToID_bench/xffl/models/llama3.1-8b-original",
    safe_serialization=True,
)
