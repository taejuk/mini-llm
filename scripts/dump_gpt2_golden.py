"""
Generate golden inputs/logits for mini-llm GPT-2 model tests.

Usage:
    pip install torch transformers numpy
    python scripts/dump_gpt2_golden.py

Outputs:
    tests/model/input_ids.bin
    tests/model/expected_logits.bin
    tests/model/decode_token.bin
    tests/model/expected_decode_logits.bin
    tests/model/meta.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tests" / "model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = "Hello, my name is"
MODEL_NAME = "gpt2"


def main() -> None:
    torch.manual_seed(0)

    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(torch.long)

    with torch.no_grad():
        prefill_out = model(input_ids, use_cache=False)

    prefill_logits = prefill_out.logits[0, -1, :].contiguous().cpu().numpy().astype(np.float32)

    decode_token = int(prefill_logits.argmax())
    decode_input = torch.tensor([[decode_token]], dtype=torch.long)
    full_decode_ids = torch.cat([input_ids, decode_input], dim=1)

    with torch.no_grad():
        full_decode_out = model(full_decode_ids, use_cache=False)

    decode_logits = full_decode_out.logits[0, -1, :].contiguous().cpu().numpy().astype(np.float32)

    flat_input_ids = input_ids[0].contiguous().cpu().numpy().astype(np.int32)
    decode_token_arr = np.array([decode_token], dtype=np.int32)

    flat_input_ids.tofile(OUT_DIR / "input_ids.bin")
    prefill_logits.tofile(OUT_DIR / "expected_logits.bin")
    decode_token_arr.tofile(OUT_DIR / "decode_token.bin")
    decode_logits.tofile(OUT_DIR / "expected_decode_logits.bin")

    meta = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "input_ids": flat_input_ids.tolist(),
        "seq_len": int(flat_input_ids.shape[0]),
        "vocab_size": int(prefill_logits.shape[0]),
        "expected_prefill_top1": int(prefill_logits.argmax()),
        "decode_token": decode_token,
        "expected_decode_top1": int(decode_logits.argmax()),
        "dtype": "float32",
    }

    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"saved {OUT_DIR / 'input_ids.bin'} shape={flat_input_ids.shape}")
    print(f"saved {OUT_DIR / 'expected_logits.bin'} shape={prefill_logits.shape}")
    print(f"saved {OUT_DIR / 'decode_token.bin'} value={decode_token}")
    print(f"saved {OUT_DIR / 'expected_decode_logits.bin'} shape={decode_logits.shape}")
    print(f"expected_prefill_top1={int(prefill_logits.argmax())}")
    print(f"expected_decode_top1={int(decode_logits.argmax())}")


if __name__ == "__main__":
    main()
