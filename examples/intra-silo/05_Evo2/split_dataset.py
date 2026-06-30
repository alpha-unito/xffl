from pathlib import Path

from Bio.Seq import Seq
from datasets import Dataset as HFDataset
from datasets import DatasetDict

BASE_PATH: Path = Path("/beegfs/home/gmittone/xffl")


def _load_fasta_dataset(name, path, test_split: float = 0.2, seed: int = 42) -> None:
    sequences: list[str] = []
    current_sequence: list[str] = []
    reverse_sequence: list[str] = []

    with open(path / f"{name}.fasta") as f:
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

    dataset: HFDataset = HFDataset.from_dict(mapping={"sequence": sequences})
    split: DatasetDict = dataset.train_test_split(
        test_size=test_split,
        seed=seed,
        shuffle=True,
    )
    split.save_to_disk(
        dataset_dict_path=BASE_PATH / f"/examples/intra-silo/05_Evo2/dataset/{name}"
    )


for dataset_name in ["flye_canu", "klebsiella"]:
    _load_fasta_dataset(name=dataset_name, path=BASE_PATH / "dataset/genome/")
