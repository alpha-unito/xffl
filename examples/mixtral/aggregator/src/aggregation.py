"""Aggregation utilities
"""

import argparse

from tqdm import tqdm
from transformers import LlamaForCausalLM


def main(args: argparse.Namespace) -> None:  # TODO: allow more aggregation strategies
    """Aggregation of multiple models

    Performs a simple models' weights averaging.
    The aggregated model is saved in half precision (as usual with LLMs)

    :param args: Command line arguments
    :type args: argparse.Namespace
    """
    print("Starting aggreagation...")
    buffer_model = None
    state_dict_list = []
    for index, model_path in enumerate(args.model):
        print(f"\t Loading model {index+1} of {len(args.model)} models...")
        buffer_model = LlamaForCausalLM.from_pretrained(model_path)
        state_dict_list.append(buffer_model.state_dict())

    def average_parameters(values):
        return sum(values) / len(values)

    print(
        f"Averaging {len(state_dict_list[0])} layers of {len(state_dict_list)} models..."
    )
    for key in tqdm(state_dict_list[0]):
        state_dict_list[0][key] = average_parameters(
            [state_dict[key] for state_dict in state_dict_list]
        )

    print("Loading the new set of weights...")
    buffer_model.load_state_dict(state_dict_list[0])
    print(f"Saving the aggregated model to {args.outname}")
    buffer_model.half().save_pretrained(args.outname, safe_serialization=False)
    print("Done!")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
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
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print("Interrupted!")
    pass
