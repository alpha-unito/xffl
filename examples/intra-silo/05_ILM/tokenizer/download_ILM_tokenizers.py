from pathlib import Path

from transformers import AutoTokenizer

data_dir: Path = Path("/beegfs/home/gmittone/ILM/ILM/tokenizer")
tokenizers_list: Path = data_dir / "ILM_tokenizers"

with open(tokenizers_list, "r") as file:
    for tokenizers_name in file.readlines():
        tokenizers_name: str = tokenizers_name.strip()
        tokenizers_path: Path = data_dir / tokenizers_name

        print(f"Downloading {tokenizers_name} to {tokenizers_path}")
        AutoTokenizer.from_pretrained(tokenizers_name).save_pretrained(tokenizers_path)
