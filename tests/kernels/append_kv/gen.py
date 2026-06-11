import json
from pathlib import Path

import numpy as np

out = Path(__file__).parent

SEQ = 6
D = 768
BLOCK_SIZE = 16

token_to_block = np.array([3, 3, 7, 7, 7, 9], dtype=np.int32)
token_to_offset = np.array([0, 1, 0, 1, 2, 4], dtype=np.int32)

buf_qkv = np.zeros((SEQ, 3 * D), dtype=np.float32)

for t in range(SEQ):
    # Q는 append_kv 검증에는 필요 없지만, layout 확인용으로 채움
    buf_qkv[t, 0:D] = 10_000 + t * 100 + np.arange(D, dtype=np.float32)
    buf_qkv[t, D:2*D] = 20_000 + t * 100 + np.arange(D, dtype=np.float32)
    buf_qkv[t, 2*D:3*D] = 30_000 + t * 100 + np.arange(D, dtype=np.float32)

expected_k = buf_qkv[:, D:2*D].copy()
expected_v = buf_qkv[:, 2*D:3*D].copy()

buf_qkv.tofile(out / "buf_qkv.bin")
token_to_block.tofile(out / "token_to_block.bin")
token_to_offset.tofile(out / "token_to_offset.bin")
expected_k.tofile(out / "expected_k.bin")
expected_v.tofile(out / "expected_v.bin")

(out / "meta.json").write_text(json.dumps({
    "seq": SEQ,
    "D": D,
    "block_size": BLOCK_SIZE,
    "dtype": "float32"
}, indent=2))