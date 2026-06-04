"""
export_weights.py
GPT-2 small 가중치를 HuggingFace에서 받아서
gpt2_model / gpt2_wmma 가 읽는 .bin 파일로 저장.

사용법:
    pip install transformers torch
    python export_weights.py          # → weights/ 폴더에 저장
"""

import os
import numpy as np
import torch
from transformers import GPT2Model

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))   # weights/ 폴더 자체

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

    # Attention — HuggingFace는 c_attn 으로 QKV 통합 저장
    # c_attn.weight : [768, 2304] (column-major처럼 보이지만 실제 [in, 3*out])
    # 우리 포맷: qkv_w [3*D, D], 즉 [2304, 768]
    qkv_w = sd[f"{p}.attn.c_attn.weight"].numpy()   # [768, 2304]
    save(qkv_w.T,                                    f"qkv_w_{l}.bin")  # [2304, 768]
    save(sd[f"{p}.attn.c_attn.bias"].numpy(),        f"qkv_b_{l}.bin")  # [2304]

    out_w = sd[f"{p}.attn.c_proj.weight"].numpy()   # [768, 768]
    save(out_w.T,                                    f"out_w_{l}.bin")  # [768, 768]
    save(sd[f"{p}.attn.c_proj.bias"].numpy(),        f"out_b_{l}.bin")

    # FFN
    fc1_w = sd[f"{p}.mlp.c_fc.weight"].numpy()      # [768, 3072]
    save(fc1_w.T,                                    f"fc1_w_{l}.bin")  # [3072, 768]
    save(sd[f"{p}.mlp.c_fc.bias"].numpy(),           f"fc1_b_{l}.bin")

    fc2_w = sd[f"{p}.mlp.c_proj.weight"].numpy()    # [3072, 768]
    save(fc2_w.T,                                    f"fc2_w_{l}.bin")  # [768, 3072]
    save(sd[f"{p}.mlp.c_proj.bias"].numpy(),         f"fc2_b_{l}.bin")

# ── Final LN ───────────────────────────────────────────────────
save(sd["ln_f.weight"].numpy(), "ln_f_w.bin")
save(sd["ln_f.bias"].numpy(),   "ln_f_b.bin")

print(f"\nAll weights saved to: {SAVE_DIR}")
