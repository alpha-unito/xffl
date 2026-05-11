import argparse
import os
from itertools import chain

from datasets import load_dataset
from transformers import AutoTokenizer


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1000000)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--tokenizer_path", type=str, default="models/llama3.1-8b")
    parser.add_argument("--output_dir", type=str, default="data/clean_mc4_it/train")
    return parser.parse_args()


# =========================================================
# TOKENIZATION
# =========================================================
def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=False,
    )


# =========================================================
# PACKING (THE IMPORTANT PART)
# =========================================================
def group_texts(examples, block_size):
    concatenated = list(chain.from_iterable(examples["input_ids"]))
    total_length = len(concatenated)

    total_length = (total_length // block_size) * block_size

    input_ids = [
        concatenated[i : i + block_size] for i in range(0, total_length, block_size)
    ]

    return {
        "input_ids": input_ids,
        "labels": input_ids.copy(),
    }


# =========================================================
# MAIN
# =========================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset gsarti/clean_mc4_it (tiny)...")
    dataset = load_dataset("gsarti/clean_mc4_it", "tiny", trust_remote_code=True)
    dataset = dataset["train"]

    if args.num_samples > len(dataset):
        args.num_samples = len(dataset)
    dataset = dataset.select(range(args.num_samples))

    print(f"Using {len(dataset)} samples")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------
    # STEP 1 — TOKENIZATION
    # -----------------------------------------------------
    print("Tokenizing dataset...")
    tokenized = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.block_size),
        batched=True,
        batch_size=args.batch_size,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # -----------------------------------------------------
    # STEP 2 — CONCAT + CHUNK FIXED BLOCKS
    # -----------------------------------------------------
    print("Packing tokens into fixed-length blocks...")
    lm_dataset = tokenized.map(
        lambda x: group_texts(x, args.block_size),
        batched=True,
        batch_size=args.batch_size,
        desc="Packing",
    )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------
    print("Saving packed dataset to disk...")
    lm_dataset.save_to_disk(args.output_dir)

    print("Done!")
    print(f"Dataset ready for LLM training: {args.output_dir}")


if __name__ == "__main__":
    main()
