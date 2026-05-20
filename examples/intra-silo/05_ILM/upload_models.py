# from transformers import AutoModelForCausalLM
# import torch
import ast
import os
import re
from pathlib import Path

from tqdm import tqdm

BASE_PATH = Path("/beegfs/home/gmittone/ILM/ILM/output")
file_list = os.listdir(BASE_PATH)


def flatten_lang_list(s: str) -> str:
    match = re.search(r"\[(.*?)\]", s)
    if not match:
        return s

    list_str = "[" + match.group(1) + "]"
    items = ast.literal_eval(list_str)
    return s.replace(list_str, "-".join(items))


for model_path in tqdm(sorted(file_list, reverse=True)):
    if model_path != "logs":
        # print(f"Loading model {model_path}")
        # model = AutoModelForCausalLM.from_pretrained(
        #     BASE_PATH / model_path,
        #     use_cache=False,
        #     local_files_only=True,
        #     attn_implementation="flash_attention_2",
        #     dtype=torch.bfloat16,
        #     use_safetensors=True,
        # )

        repo_name = f"interlinguistic-language-modeling/{flatten_lang_list(model_path)}"
        # print(f"Pushing to the hub as {repo_name}")
        print(repo_name)
        # model.push_to_hub(repo_name)
