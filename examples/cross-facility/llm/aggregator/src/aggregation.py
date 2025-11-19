"""Aggregation utility"""

import argparse

import torch
from tqdm import tqdm
from transformers import AutoModel


def main(args: argparse.Namespace) -> None:
    """Aggregation of multiple models

    Performs a simple models' weights averaging.
    The aggregated model is saved in half precision (as usual with LLMs)

    :param args: Command line arguments
    :type args: argparse.Namespace
    """
    print("Loading base model...")
    base_model: AutoModel = AutoModel.from_pretrained(args.base_model)
    optimizer: torch.optim.SGD = torch.optim.SGD(
        base_model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        nesterov=True,
    )
    try:
        print("Loading optimizer state...")
        optimizer.load_state_dict(torch.load(args.optimizer))
    except Exception as e:
        print(e)

    print("Starting aggregation...")
    with torch.no_grad():
        grads = [torch.zeros_like(p) for p in base_model.parameters()]
        num_models = len(args.model)

        for index, name in enumerate(args.model):
            print(f"\t Loading model {index+1} of {num_models} models...")
            local = AutoModel.from_pretrained(name)
            print("Calculating pseudo-gradient...")
            for g, p_global, p_local in tqdm(
                zip(grads, base_model.parameters(), local.parameters())
            ):
                # g_i = M_t - M_i
                g.add_(p_global.data - p_local.data)
            del local

        print("Averaging gradients...")
        for g in tqdm(grads):
            g.div_(num_models)

        print("Applying gradients...")
        for p, g in tqdm(zip(base_model.parameters(), grads)):
            p.grad = g.clone()

    print("Optimizer step...")
    optimizer.step()
    optimizer.zero_grad()

    print(f"Saving the aggregated model and the optimizer state to {args.outname}...")
    base_model.half().save_pretrained(args.outname, safe_serialization=False)
    torch.save(optimizer.state_dict(), args.optimizer)

    print("Done!")


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
