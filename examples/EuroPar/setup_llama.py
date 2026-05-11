import os

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"
OUTPUT_DIR = "./models/llama3.1-8b"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    print("Downloading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="cpu"
    )

    print("Saving locally in safetensors format...")
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Done! Model saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
