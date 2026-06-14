"""
export_weights.py
GPT-2 small 가중치를 HuggingFace에서 받아서
mini-llm이 읽는 weights/*.bin 파일로 저장.

사용법:
    pip install transformers torch numpy
    python scripts/export_weights.py          # → weights/ 폴더에 저장
"""

import os
import numpy as np
import torch
from transformers import GPT2Model

SAVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "weights"
)
os.makedirs(SAVE_DIR, exist_ok=True)


def save(arr: np.ndarray, name: str):
    path = os.path.join(SAVE_DIR, name)
    arr.astype(np.float32).tofile(path)
    print(f"  saved {name:40s}  shape={arr.shape}")


print("Loading GPT-2 small from HuggingFace...")
model = GPT2Model.from_pretrained("gpt2")
sd = model.state_dict()

# ── Embedding ──────────────────────────────────────────────────
save(sd["wte.weight"].numpy(),           "wte.bin")   # [50257, 768]
save(sd["wpe.weight"].numpy(),           "wpe.bin")   # [1024,  768]

# ── Per-layer ──────────────────────────────────────────────────
for l in range(12):
    p = f"h.{l}"

    # Layer Norm
    save(sd[f"{p}.ln_1.weight"].numpy(), f"ln1_w_{l}.bin")
    save(sd[f"{p}.ln_1.bias"].numpy(),   f"ln1_b_{l}.bin")
    save(sd[f"{p}.ln_2.weight"].numpy(), f"ln2_w_{l}.bin")
    save(sd[f"{p}.ln_2.bias"].numpy(),   f"ln2_b_{l}.bin")

    # Attention
    # HF Conv1D weight is used as x @ weight + bias.
    # mini-llm linear() also computes A[M,K] @ B[K,N] + bias[N], so keep
    # the HuggingFace matrix layout as-is. Do NOT transpose here.
    save(sd[f"{p}.attn.c_attn.weight"].numpy(), f"qkv_w_{l}.bin")   # [768, 2304]
    save(sd[f"{p}.attn.c_attn.bias"].numpy(),   f"qkv_b_{l}.bin")   # [2304]

    save(sd[f"{p}.attn.c_proj.weight"].numpy(), f"out_w_{l}.bin")   # [768, 768]
    save(sd[f"{p}.attn.c_proj.bias"].numpy(),   f"out_b_{l}.bin")

    # FFN
    save(sd[f"{p}.mlp.c_fc.weight"].numpy(),    f"fc1_w_{l}.bin")   # [768, 3072]
    save(sd[f"{p}.mlp.c_fc.bias"].numpy(),      f"fc1_b_{l}.bin")

    save(sd[f"{p}.mlp.c_proj.weight"].numpy(),  f"fc2_w_{l}.bin")   # [3072, 768]
    save(sd[f"{p}.mlp.c_proj.bias"].numpy(),    f"fc2_b_{l}.bin")

# ── Final LN ───────────────────────────────────────────────────
save(sd["ln_f.weight"].numpy(), "ln_f_w.bin")
save(sd["ln_f.bias"].numpy(),   "ln_f_b.bin")

print(f"\nAll weights saved to: {SAVE_DIR}")
