import pkgutil
from typing import List, Optional, Tuple

import torch
import yaml
from StripedHyena2 import StripedHyena
from torch import nn
from vortex.model.generation import generate as vortex_generate
from vortex.model.tokenizer import CharLevelTokenizer
from vortex.model.utils import dotdict

CHECKPOINT_PATH: str = (
    "/beegfs/home/gmittone/xffl/examples/intra-silo/05_Evo2/checkpoint/epoch_0/checkpoint.pt"
)


# Generation method
def generate(
    model,
    tokenizer,
    prompt_seqs: List[str],
    n_tokens: int = 500,
    temperature: float = 1.0,
    top_k: int = 4,
    top_p: float = 1.0,
    batched: bool = True,
    cached_generation: bool = True,
    verbose: int = 1,
    force_prompt_threshold: Optional[int] = None,
) -> Tuple[List[str], List[float]]:
    """
    Generate sequences from a list of prompts.

    force_prompt_threshold: If specified, avoids OOM errors through teacher forcing if the prompt is longer than this threshold.

    If force_prompt_threshold is none, sets default assuming 1xH100 (evo2_7b) and 2xH100 (evo2_40b) to help avoid OOM errors.
    """

    with torch.no_grad():
        output = vortex_generate(
            prompt_seqs=prompt_seqs,
            model=model,
            tokenizer=tokenizer,
            n_tokens=n_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            batched=batched,
            cached_generation=cached_generation,
            verbose=verbose,
            force_prompt_threshold=force_prompt_threshold,
        )
        return output  # type: ignore


# Instantiate standard Evo2 1B base
evo2_config = yaml.safe_load(pkgutil.get_data("evo2.utils", "configs/evo2-1b-8k.yml"))  # type: ignore
evo2_config = dotdict(evo2_config)  # type: ignore
evo2_config.use_fp8_input_projections = False  # type: ignore

model: nn.Module = StripedHyena(evo2_config, generation=True)

# Load the xFFL checkpoint weights into the Evo2 instance
model.custom_load_state_dict(
    torch.load(
        f=CHECKPOINT_PATH,
        weights_only=False,
    ),
    strict=False,
)

model.to(device="cuda:0", dtype=torch.bfloat16)

# Create the tokenizer
vocab_size: int = 512
tokenizer: CharLevelTokenizer = CharLevelTokenizer(vocab_size)

# Generate output sequences
output = generate(
    model=model,
    tokenizer=tokenizer,
    prompt_seqs=["ACGT"],
    n_tokens=500,
    temperature=1.0,
    top_k=4,
)

print(output.sequences[0])  # type: ignore
