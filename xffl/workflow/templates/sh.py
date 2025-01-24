def get_aggregate():
    return """import argparse
import math
import random
import os

from pathlib import Path
from statistics import mean
from string import ascii_lowercase


NUM_FILE_DIRECTORY = 2


def _create_file(path, size):
    # text = "".join(
    #     [random.choice(ascii_lowercase) for _ in range(random.randint(10, 100))]
    # )
    with open(path, "wb") as fd:
        fd.seek(size - 1)
        fd.write(b"\\0")


def _get_type_string(path):
    if os.path.isfile(path):
        return "File"
    elif os.path.isdir(path):
        return "Directory"
    else:
        return "Unknown"


def get_size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    else:
        total_size = 0
        for dirpath, _, filenames in os.walk(path, followlinks=True):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size


def main(args):
    random.seed(args.seed)
    model_name = args.outname
    model_avg_size = mean(get_size(path) for path in args.model)
    print(f"Avg model size: {model_avg_size} bytes")
    if all(os.path.isdir(path) for path in args.model):
        os.mkdir(model_name)
        sizes = [
            math.floor(model_avg_size / NUM_FILE_DIRECTORY)
            for _ in range(NUM_FILE_DIRECTORY)
        ]
        files_avg_size = mean(sizes)
        # Shuffle file sizes to have files with different sizes
        for _ in range(10):
            i = random.randint(0, NUM_FILE_DIRECTORY - 1)
            j = random.randint(0, NUM_FILE_DIRECTORY - 1)
            quantity = random.randint(1, files_avg_size - 1)
            if sizes[i] > quantity:
                sizes[i] -= quantity
                sizes[j] += quantity
        for i, size in enumerate(sizes):
            _create_file(os.path.join(model_name, f"file{i}.txt"), size=size)
    elif all(os.path.isfile(path) for path in args.model):
        if not Path(model_name).suffix:
            model_name = Path(model_name).with_suffix(".txt")
        _create_file(model_name, size=model_avg_size)
    else:
        raise Exception(
            f"Inputs are all differents. "
            f"Accepted only all files or all directories. "
            f"Instead got {[_get_type_string(path) for path in args.model]}"
        )
    print(f"New model size: {get_size(model_name)} bytes")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model",
            action="append",
            help="Input path of a local model",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--outname",
            help="Output path of the aggregated model",
            type=str,
            required=True,
        )
        parser.add_argument("--seed", help="Random generator seed", type=int, default=8)
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print("Interrupted!")
    pass
"""
