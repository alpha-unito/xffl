from pathlib import Path

from datasets import load_dataset

data_dir: Path = Path("/beegfs/home/gmittone/ILM/ILM/data")
datasets_list: Path = data_dir / "ILM_datasets"

with open(datasets_list, "r") as file:
    for dataset_name in file.readlines():
        dataset_name: str = dataset_name.strip()
        dataset_path: Path = data_dir / dataset_name

        print(f"Downloading {dataset_name} to {dataset_path}")
        load_dataset(dataset_name).save_to_disk(dataset_path)
