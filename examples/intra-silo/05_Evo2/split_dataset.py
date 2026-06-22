from pathlib import Path

from Bio.Seq import Seq
from datasets import Dataset as HFDataset
from datasets import DatasetDict


def _load_fasta_dataset(
    path: Path,
    test_split: float = 0.2,
    seed: int = 42,
    name="",
) -> DatasetDict:
    sequences: list[str] = []
    current_sequence: list[str] = []
    reverse_sequence: list[str] = []

    with open(path + name + ".fasta") as f:
        for line in f:
            line: str = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if current_sequence:
                    sequences.append("".join(current_sequence))
                    sequences.append("".join(reverse_sequence))
                    current_sequence = []
                    reverse_sequence = []
            else:
                current_sequence.append(line)
                reverse_sequence.append(str(Seq(line).reverse_complement()))

        if current_sequence:
            sequences.append("".join(current_sequence))

    dataset = HFDataset.from_dict({"sequence": sequences})

    split: DatasetDict = dataset.train_test_split(
        test_size=test_split,
        seed=seed,
        shuffle=True,
    )

    split.save_to_disk(
        f"/beegfs/home/gmittone/xffl/examples/intra-silo/05_Evo2/dataset/{name}"
    )

    return split


_load_fasta_dataset("/beegfs/home/gmittone/xffl/dataset/genome/", name="flye_canu")
_load_fasta_dataset("/beegfs/home/gmittone/xffl/dataset/genome/", name="klebsiella")
