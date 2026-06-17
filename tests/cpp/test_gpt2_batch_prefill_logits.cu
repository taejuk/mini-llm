"""
Generate golden files for batched GPT-2 prefill logits test.

Usage:
    pip install torch transformers numpy
    python scripts/dump_gpt2_batch_prefill_golden.py

Outputs:
    tests/model/batch_prefill_input_ids.bin
    tests/model/batch_prefill_prompt_lens.bin
    tests/model/batch_prefill_expected_logits.bin
    tests/model/batch_prefill_meta.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tests" / "model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "gpt2"

# Valid GPT-2 token ids.
# We intentionally use different lengths, including KV block boundary cases:
# 15, 16, 17 where DEFAULT_KV_BLOCK_SIZE = 16.
BASE_TOKENS = [
    15496, 11, 616, 1438, 318,
    257, 845, 922, 640, 13,
    198, 464, 3139, 286, 4881,
    318, 262, 1748, 13, 198,
    1026, 373, 281, 1593, 640,
]


def make_tokens(length: int, shift: int) -> list[int]:
    return [
        BASE_TOKENS[(i + shift) % len(BASE_TOKENS)]
        for i in range(length)
    ]


CASES = [
    {"request_id": 0, "name": "len5", "tokens": make_tokens(5, 0)},
    {"request_id": 1, "name": "len15", "tokens": make_tokens(15, 3)},
    {"request_id": 2, "name": "len16", "tokens": make_tokens(16, 7)},
    {"request_id": 3, "name": "len17", "tokens": make_tokens(17, 11)},
]


def main() -> None:
    torch.manual_seed(0)

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()

    expected_logits = []

    with torch.no_grad():
        for case in CASES:
            input_ids = torch.tensor(
                [case["tokens"]],
                dtype=torch.long,
            )

            out = model(input_ids, use_cache=False)
            logits = (
                out.logits[0, -1, :]
                .contiguous()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            expected_logits.append(logits)

            case["prompt_len"] = len(case["tokens"])
            case["expected_prefill_top1"] = int(logits.argmax())

    flat_input_ids = np.array(
        [tok for case in CASES for tok in case["tokens"]],
        dtype=np.int32,
    )

    prompt_lens = np.array(
        [case["prompt_len"] for case in CASES],
        dtype=np.int32,
    )

    expected_logits_np = np.stack(expected_logits, axis=0).astype(np.float32)

    flat_input_ids.tofile(OUT_DIR / "batch_prefill_input_ids.bin")
    prompt_lens.tofile(OUT_DIR / "batch_prefill_prompt_lens.bin")
    expected_logits_np.tofile(OUT_DIR / "batch_prefill_expected_logits.bin")

    meta = {
        "model": MODEL_NAME,
        "batch_size": len(CASES),
        "vocab_size": int(expected_logits_np.shape[1]),
        "dtype": "float32",
        "layout": {
            "input_ids": "flattened int32 [sum(prompt_lens)]",
            "prompt_lens": "int32 [batch_size]",
            "expected_logits": "float32 [batch_size, vocab_size]",
        },
        "cases": CASES,
    }

    (OUT_DIR / "batch_prefill_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print(f"saved {OUT_DIR / 'batch_prefill_input_ids.bin'} shape={flat_input_ids.shape}")
    print(f"saved {OUT_DIR / 'batch_prefill_prompt_lens.bin'} shape={prompt_lens.shape}")
    print(f"saved {OUT_DIR / 'batch_prefill_expected_logits.bin'} shape={expected_logits_np.shape}")
    print(f"saved {OUT_DIR / 'batch_prefill_meta.json'}")

    for case in CASES:
        print(
            f"request_id={case['request_id']} "
            f"name={case['name']} "
            f"prompt_len={case['prompt_len']} "
            f"expected_top1={case['expected_prefill_top1']}"
        )


if __name__ == "__main__":
    main()