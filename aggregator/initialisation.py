import sys
import argparse
import os
import torch

from transformers import AutoModelForCausalLM
from torch import nn

def get_weight_norms(model: nn.Module, verbose: bool = False) -> None:
    """
    Prints the L1 norm of the weights of each module in the given PyTorch model.

    Args:
    model (nn.Module): The PyTorch model whose weight norms are to be printed.

    Returns:
    None
    """
    total_weight_norm: float = 0.0
    for name, module in model.named_modules():
        # Check if the module has the 'weight' attribute
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
            # Calculate the L1 norm of the weights
            w_norm: float = module.weight.norm(1).item()
            total_weight_norm += w_norm
            if verbose:
                print(f"Norm of weights in module {name}: {w_norm}")
    return total_weight_norm

def reinitialize_weights(model, 
                         std: float = 0.0002,  # 0.02 ref: hailey S doesn't recommend this huge value! 
                         ) -> None:
    """
    
    From cs197, we choose std = 0.02 because of these two links:
    Why we chose 0.02 for standard deviation:
    https://github.com/huggingface/transformers/blob/772307be7649e1333a933cfaa229dc0dec2fd331/src/transformers/models/llama/modeling_llama.py#L858
    https://github.com/huggingface/transformers/blob/772307be7649e1333a933cfaa229dc0dec2fd331/src/transformers/models/llama/configuration_llama.py#L127
    Default is set to 0.02 in source code (see line 858 of the first link, and 127 of hte second link)
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # nn.init.normal_(module.weight, mean=0, std=0.02)
            nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

def main(args):
	print("Starting initialisation...")
	for index, model_path in enumerate(args.model):
		print(f"\t Loading model {index+1} of {len(args.model)} models...")
		model = AutoModelForCausalLM.from_pretrained(model_path)

		total_weight_norm_before_reinit = get_weight_norms(model)
		print(f"Total weight norm (before): {total_weight_norm_before_reinit}")
		reinitialize_weights(model)
		total_weight_norm_after_reinit = get_weight_norms(model)
		print(f"Total weight norm (after): {total_weight_norm_after_reinit=}")

		model.half().save_pretrained(args.outname, safe_serialization=False)

if __name__ == "__main__":
	try:
		parser = argparse.ArgumentParser()
		parser.add_argument('-m', '--model', action='append', help='Input path of a local model', type=str, required=True)
		parser.add_argument('-o', '--outname', help='Output path of the aggregated model', type=str, required=True)
		args = parser.parse_args()
		main(args)
	except KeyboardInterrupt:
		print("Interrupted!")
	pass