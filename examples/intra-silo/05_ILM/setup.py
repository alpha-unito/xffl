# =============================================================
# FILE 2 — DATA PIPELINE + LLAMA-STYLE 100 M MODEL
# =============================================================
# Expects the tokenizer directory produced by 01_train_tokenizer.py:
#   llama_ita_tokenizer/
#
# What this file does
#   • Loads the full ilm_ita dataset (5 M rows)
#   • Builds an IterableDataset that tokenises on-the-fly and
#     packs examples into fixed-length blocks (no padding)
#   • Instantiates a Llama-style causal LM (~100 M parameters)
#     from scratch using LlamaForCausalLM + LlamaConfig
#   • Creates an AdamW optimiser with a cosine LR schedule
#     (standard Llama pre-training recipe)
#   • Exports everything in a training-ready state
#     (the actual training loop is left to you)
#
# Install
#   pip install transformers datasets accelerate torch
# =============================================================

import os

# ------------------------------------------------------------------
# 0. IMPORTS
# ------------------------------------------------------------------
import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ------------------------------------------------------------------
# 1. HYPER-PARAMETERS  (edit here to tune)
# ------------------------------------------------------------------
TOKENIZER_DIR = (
    "/beegfs/home/gmittone/xffl/examples/intra-silo/05_ILM/ilm_ita"  # output of file 1
)
DATASET_NAME = "interlinguistic-language-modeling/ilm_ita"
TEXT_COLUMN = "text"

BLOCK_SIZE = 1024  # context window (tokens)
BATCH_SIZE = 8  # per-device micro-batch (adjust to VRAM)
NUM_WORKERS = 0  # set > 0 once you confirm correctness

# Optimiser / schedule
LEARNING_RATE = 3e-4  # peak LR (Llama pre-training default)
WEIGHT_DECAY = 0.1
BETA1, BETA2 = 0.9, 0.95
WARMUP_STEPS = 2_000
TOTAL_STEPS = 100_000  # adjust to your compute budget

BATCH_SIZE_TOKENIZATION = 4096

# ------------------------------------------------------------------
# 2. LOAD TOKENIZER
# ------------------------------------------------------------------
print(f"Loading tokenizer from '{TOKENIZER_DIR}' …")
tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)

VOCAB_SIZE = len(tokenizer)
BOS_ID = tokenizer.bos_token_id
EOS_ID = tokenizer.eos_token_id
PAD_ID = tokenizer.pad_token_id

print(f"  vocab_size={VOCAB_SIZE}, bos={BOS_ID}, eos={EOS_ID}, pad={PAD_ID}")

# ------------------------------------------------------------------
# 3. LOAD FULL DATASET
# ------------------------------------------------------------------
print(f"Loading dataset '{DATASET_NAME}' …")
dataset = load_dataset(DATASET_NAME, split="train").shuffle(seed=42)
print(f"  Total examples: {len(dataset):,}")


# ------------------------------------------------------------------
# 4. PRECOMPUTE TOKEN STREAM
# ------------------------------------------------------------------
# print("Tokenizing dataset...")

# _encode_batch = tokenizer._tokenizer.encode_batch

# all_tokens = []

# pbar = tqdm(
#     total=len(dataset),
#     desc="Tokenizing",
#     unit="docs",
# )

# with open("tokens.bin", "wb") as f:
#     for start in tqdm(range(0, len(dataset), BATCH_SIZE_TOKENIZATION)):
#         end = min(start + BATCH_SIZE_TOKENIZATION, len(dataset))

#         texts = [t for t in dataset[start:end][TEXT_COLUMN] if t]

#         encodings = tokenizer._tokenizer.encode_batch(texts)

#         for enc in encodings:
#             np.asarray(enc.ids, dtype=np.int32).tofile(f)

#         pbar.update(end - start)

# pbar.close()

# all_tokens = np.asarray(all_tokens, dtype=np.int32)

# print(f"Total tokens: {len(all_tokens):,}")


# ------------------------------------------------------------------
# 5. MAP-STYLE DATASET
# ------------------------------------------------------------------
class LlamaDataset(Dataset):
    def __init__(
        self,
        token_file: str,
        block_size: int,
        dtype=np.int32,
    ):
        self.block_size = block_size

        self.tokens = np.memmap(
            token_file,
            mode="r",
            dtype=dtype,
        )

        self.num_blocks = len(self.tokens) // block_size

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size

        ids = torch.from_numpy(self.tokens[start:end].astype(np.int64))

        return {
            "input_ids": ids[:-1],
            "labels": ids[1:],
        }

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]


train_dataset = LlamaDataset(
    token_file="/beegfs/home/gmittone/xffl/examples/intra-silo/05_ILM/tokens.bin",
    block_size=BLOCK_SIZE,
)

print(f"Samples: {len(train_dataset):,}")

# ------------------------------------------------------------------
# 7. DATALOADER
# ------------------------------------------------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

# ------------------------------------------------------------------
# 8. LLAMA MODEL CONFIG — ~100 M PARAMETERS
# ------------------------------------------------------------------
# Architecture analysis (Llama-style):
#
#   Component                              Parameters
#   ─────────────────────────────────────────────────
#   Token embeddings  32 000 × 1024        ~32.8 M
#   12 × TransformerBlock:
#     Self-attention  4 × (1024² + bias)   ~4.2 M / block
#     SwiGLU FFN      3 × (1024 × 2816)    ~8.7 M / block
#     RMSNorm × 2     2 × 1024             negligible
#   12 blocks total                        ~155 M  ← too many
#
#   Tuned: hidden=960, n_layers=10, intermediate=2560
#   Token embeddings  32 000 × 960         ~30.7 M
#   10 × block:
#     Attention       4 × (960²)           ~3.7 M / block
#     FFN SwiGLU      3 × (960 × 2560)     ~7.4 M / block
#   10 blocks                              ~110 M
#   LM head (weight-tied to embed)         0
#   ─────────────────────────────────────────────────
#   Total                                  ≈ 141 M
#
#   Final tuning → hidden=768, n_layers=12, intermediate=2048:
#   Token embed  16 000 × 768             ~12.3 M
#   12 × block   (768² × 4 + 768 × 2048 × 3) ≈ 6.0 M / block
#   12 blocks                             ~72.0 M
#   Total                                 ≈ 84 M
#
#   Compensate for smaller embed: bump to hidden=832, intermediate=2240
#   Token embed  16 000 × 832             ~13.3 M
#   12 × block   (832² × 4 + 832 × 2240 × 3) ≈ 7.0 M / block
#   12 blocks                             ~84.0 M
#   Total                                 ≈ 97 M  ✓  (~100 M)
#
# Llama-specific choices vs vanilla Transformer:
#   • RMSNorm instead of LayerNorm (no bias, faster)
#   • RoPE positional encodings (no learned embeddings)
#   • SwiGLU activation with gated FFN
#   • No attention bias (bias=False throughout)
#   • Pre-norm (norm before attention/FFN, not after)

config = LlamaConfig(
    vocab_size=VOCAB_SIZE,  # 32 000
    # --- dimensions ---
    hidden_size=832,  # d_model (bumped from 768 to compensate for smaller embed table)
    intermediate_size=2240,  # FFN inner dim (~2.7 × hidden, SwiGLU)
    num_hidden_layers=12,  # transformer depth
    # --- attention ---
    num_attention_heads=8,  # MHA heads  (832 / 8 = 104 per head)
    num_key_value_heads=8,  # GQA: set < num_attention_heads for MQA/GQA
    # --- Llama defaults ---
    hidden_act="silu",  # SwiGLU uses SiLU gating
    max_position_embeddings=BLOCK_SIZE,  # RoPE base length — 1024
    rms_norm_eps=1e-5,
    rope_theta=10_000.0,  # RoPE base frequency (original Llama)
    # --- regularisation ---
    attention_dropout=0.0,  # Llama pre-training uses 0
    # --- special tokens ---
    bos_token_id=BOS_ID,
    eos_token_id=EOS_ID,
    pad_token_id=PAD_ID,
    # --- weight tying ---
    tie_word_embeddings=True,  # LM head shares weights with embed table
)

# ------------------------------------------------------------------
# 9. INSTANTIATE MODEL FROM SCRATCH
# ------------------------------------------------------------------
print("Initialising Llama model from scratch …")
model = LlamaForCausalLM(config)

# Xavier / Kaiming init is applied by __init__; no extra step needed.

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters    : {total_params:,}  (~{total_params / 1e6:.1f} M)")
print(f"  Trainable parameters: {trainable_params:,}")

# ------------------------------------------------------------------
# 10. OPTIMISER — AdamW (Llama pre-training recipe)
# ------------------------------------------------------------------
# Separate weight-decay and no-decay parameter groups (standard practice:
# don't decay biases, norms, or embeddings).
decay_params = []
no_decay_params = []

NO_DECAY_NAMES = ("bias", "norm", "embed")  # heuristic substring match

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if any(nd in name for nd in NO_DECAY_NAMES):
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=LEARNING_RATE,
    betas=(BETA1, BETA2),
    eps=1e-8,
    fused=torch.cuda.is_available(),  # faster fused kernel when on GPU
)
print(
    f"Optimiser: AdamW  lr={LEARNING_RATE}  wd={WEIGHT_DECAY}  betas=({BETA1},{BETA2})"
)

# ------------------------------------------------------------------
# 11. LR SCHEDULER — cosine decay with linear warm-up  (Llama style)
# ------------------------------------------------------------------
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=TOTAL_STEPS,
)
print(
    f"Scheduler: cosine w/ {WARMUP_STEPS} warmup steps over {TOTAL_STEPS} total steps"
)

# ------------------------------------------------------------------
# 12. DEVICE PLACEMENT
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model moved to: {device}")
