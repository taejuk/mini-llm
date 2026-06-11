import json
from pathlib import Path

import numpy as np

out = Path(__file__).parent

SEQ = 7
VOCAB = 64
MAX_SEQ = 32
D = 768

tokens = np.array([3, 5, 7, 5, 11, 13, 17], dtype=np.int32)
pos = np.arange(SEQ, dtype=np.int32)

wte = np.zeros((VOCAB, D), dtype=np.float32)
wpe = np.zeros((MAX_SEQ, D), dtype=np.float32)

for t in range(VOCAB):
    for d in range(D):
        wte[t, d] = t * 0.01 + d * 0.0001

for p in range(MAX_SEQ):
    for d in range(D):
        wpe[p, d] = p * 0.001 + d * 0.00001

expected = wte[tokens] + wpe[pos]
expected = expected.astype(np.float32)

tokens.tofile(out / "tokens.bin")
pos.tofile(out / "pos.bin")
wte.tofile(out / "wte.bin")
wpe.tofile(out / "wpe.bin")
expected.tofile(out / "expected.bin")

(out / "meta.json").write_text(json.dumps({
    "seq": SEQ,
    "vocab": VOCAB,
    "max_seq": MAX_SEQ,
    "D": D,
    "dtype": "float32"
}, indent=2))