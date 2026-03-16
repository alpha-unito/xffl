"""Aggregation utility"""

import argparse

import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM


def main(args: argparse.Namespace) -> None:
    """Aggregation of multiple models

    Performs a simple models' weights averaging.
    The aggregated model is saved in half precision (as usual with LLMs)

    :param args: Command line arguments
    :type args: argparse.Namespace
    """
    print("Loading base model...")
    base_model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=True,
        attn_implementation="sdpa",
        dtype=torch.float32,
        device_map="cpu",
        use_safetensors=True,
    )

    optimizer = torch.optim.AdamW(
        base_model.parameters(), lr=0.1, betas=(0.9, 0.99), eps=1e-8
    )
    try:
        print("Loading optimizer state...")
        optimizer.load_state_dict(torch.load(args.optimizer))
    except Exception as e:
        print(e)

    print("Starting aggregation...")
    deltas = [torch.zeros_like(p, dtype=torch.float32) for p in base_model.parameters()]

    num_models = len(args.model)
    for index, path in enumerate(args.model):
        print(f"\t Loading model {index+1} of {num_models} models from {path}...")
        local = LlamaForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            attn_implementation="sdpa",
            dtype=torch.float32,
            device_map="cpu",
            use_safetensors=True,
        )

        for d, p_global, p_local in tqdm(
            zip(deltas, base_model.parameters(), local.parameters())
        ):
            d.add_(p_local.data - p_global.data)
        del local

    for d in deltas:
        d.div_(num_models)

    total_norm = sum(d.norm().item() for d in deltas)
    print(f"Norma dei cambiamenti: {total_norm}")

    print("Applying gradients...")
    for p, d in tqdm(zip(base_model.parameters(), deltas)):
        p.grad = -d

    print("Optimizer step...")

    torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
    total_norm = sum(p.grad.norm().item() for p in base_model.parameters())
    print(f"Norma dei gradienti: {total_norm}")

    params_before = [p.data.clone() for p in base_model.parameters()]
    optimizer.step()
    diff = sum(
        (p.data - p_before).norm().item()
        for p, p_before in zip(base_model.parameters(), params_before)
    )

    print("Total change after optimizer.step():", diff)
    optimizer.zero_grad()

    print(f"Saving the aggregated model and the optimizer state to {args.outname}...")
    base_model.to(dtype=torch.bfloat16)
    base_model.save_pretrained(args.outname, safe_serialization=True)
    torch.save(optimizer.state_dict(), args.optimizer)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-b",
            "--base-model",
            help="Previous round aggregated model",
            type=str,
            required=True,
        )

        parser.add_argument(
            "-m",
            "--model",
            action="append",
            help="Paths of the models to be aggregated",
            type=str,
            required=True,
        )

        parser.add_argument(
            "-o",
            "--outname",
            help="Path where to save to aggregated model",
            type=str,
            required=True,
        )

        parser.add_argument(
            "-opt",
            "--optimizer",
            help="Path where to save to optimizer state",
            type=str,
            required=True,
        )

        parser.add_argument(
            "-lr",
            "--learning-rate",
            help="Learning rate",
            type=float,
            default=0.7,
        )

        parser.add_argument(
            "-mt",
            "--momentum",
            help="Momentum",
            type=float,
            default=0.9,
        )

        arguments: argparse.Namespace = parser.parse_args()
        main(args=arguments)
    except KeyboardInterrupt:
        print("Interrupted!")
    pass
