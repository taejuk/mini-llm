"""
Generate golden inputs/logits for the mini-llm GPT-2 prefill model test.

Usage:
    pip install torch transformers numpy
    python scripts/dump_gpt2_golden.py

Outputs:
    tests/model/input_ids.bin
    tests/model/expected_logits.bin
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
        out = model(input_ids, use_cache=False)

    last_logits = out.logits[0, -1, :].contiguous().cpu().numpy().astype(np.float32)
    flat_input_ids = input_ids[0].contiguous().cpu().numpy().astype(np.int32)

    flat_input_ids.tofile(OUT_DIR / "input_ids.bin")
    last_logits.tofile(OUT_DIR / "expected_logits.bin")

    top1 = int(last_logits.argmax())
    meta = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "input_ids": flat_input_ids.tolist(),
        "seq_len": int(flat_input_ids.shape[0]),
        "vocab_size": int(last_logits.shape[0]),
        "expected_top1": top1,
        "dtype": "float32",
    }

    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"saved {OUT_DIR / 'input_ids.bin'} shape={flat_input_ids.shape}")
    print(f"saved {OUT_DIR / 'expected_logits.bin'} shape={last_logits.shape}")
    print(f"expected_top1={top1}")


if __name__ == "__main__":
    main()
