"""
Generate golden files for batched GPT-2 decode logits test.

Usage:
    pip install torch transformers numpy
    python scripts/dump_gpt2_batch_decode_golden.py

Outputs:
    tests/model/batch_decode_input_ids.bin
    tests/model/batch_decode_prompt_lens.bin
    tests/model/batch_decode_tokens.bin
    tests/model/batch_decode_expected_logits.bin
    tests/model/batch_decode_meta.json
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
# Use different prompt lengths, including KV block boundary cases.
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

    decode_tokens: list[int] = []
    expected_decode_logits = []

    with torch.no_grad():
        for case in CASES:
            prompt_ids = torch.tensor(
                [case["tokens"]],
                dtype=torch.long,
            )

            # 1. Prefill logits에서 greedy decode token 선택
            prefill_out = model(prompt_ids, use_cache=False)
            prefill_logits = prefill_out.logits[0, -1, :]
            decode_token = int(prefill_logits.argmax().item())

            # 2. prompt + decode_token을 넣었을 때의 last logits가
            #    C++ decode 1-step과 비교할 정답
            decode_input = torch.tensor(
                [[decode_token]],
                dtype=torch.long,
            )

            full_ids = torch.cat(
                [prompt_ids, decode_input],
                dim=1,
            )

            decode_out = model(full_ids, use_cache=False)

            logits = (
                decode_out.logits[0, -1, :]
                .contiguous()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            expected_decode_logits.append(logits)
            decode_tokens.append(decode_token)

            case["prompt_len"] = len(case["tokens"])
            case["decode_token"] = decode_token
            case["expected_decode_top1"] = int(logits.argmax())

    flat_input_ids = np.array(
        [tok for case in CASES for tok in case["tokens"]],
        dtype=np.int32,
    )

    prompt_lens = np.array(
        [case["prompt_len"] for case in CASES],
        dtype=np.int32,
    )

    decode_tokens_np = np.array(
        decode_tokens,
        dtype=np.int32,
    )

    expected_decode_logits_np = np.stack(
        expected_decode_logits,
        axis=0,
    ).astype(np.float32)

    flat_input_ids.tofile(
        OUT_DIR / "batch_decode_input_ids.bin"
    )

    prompt_lens.tofile(
        OUT_DIR / "batch_decode_prompt_lens.bin"
    )

    decode_tokens_np.tofile(
        OUT_DIR / "batch_decode_tokens.bin"
    )

    expected_decode_logits_np.tofile(
        OUT_DIR / "batch_decode_expected_logits.bin"
    )

    meta = {
        "model": MODEL_NAME,
        "batch_size": len(CASES),
        "vocab_size": int(expected_decode_logits_np.shape[1]),
        "dtype": "float32",
        "layout": {
            "input_ids": "flattened int32 [sum(prompt_lens)]",
            "prompt_lens": "int32 [batch_size]",
            "decode_tokens": "int32 [batch_size]",
            "expected_decode_logits": "float32 [batch_size, vocab_size]",
        },
        "cases": CASES,
    }

    (OUT_DIR / "batch_decode_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print(f"saved {OUT_DIR / 'batch_decode_input_ids.bin'} shape={flat_input_ids.shape}")
    print(f"saved {OUT_DIR / 'batch_decode_prompt_lens.bin'} shape={prompt_lens.shape}")
    print(f"saved {OUT_DIR / 'batch_decode_tokens.bin'} shape={decode_tokens_np.shape}")
    print(f"saved {OUT_DIR / 'batch_decode_expected_logits.bin'} shape={expected_decode_logits_np.shape}")
    print(f"saved {OUT_DIR / 'batch_decode_meta.json'}")

    for case in CASES:
        print(
            f"request_id={case['request_id']} "
            f"name={case['name']} "
            f"prompt_len={case['prompt_len']} "
            f"decode_token={case['decode_token']} "
            f"expected_decode_top1={case['expected_decode_top1']}"
        )


if __name__ == "__main__":
    main()