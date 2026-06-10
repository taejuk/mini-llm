import json
from pathlib import Path

import numpy as np

out = Path(__file__).parent

ROWS = 17
COLS = 768
EPS = 1e-5

rng = np.random.default_rng(0)

x = rng.normal(0.0, 1.0, size=(ROWS, COLS)).astype(np.float32)
gamma = rng.normal(1.0, 0.1, size=(COLS,)).astype(np.float32)
beta = rng.normal(0.0, 0.1, size=(COLS,)).astype(np.float32)

mean = x.mean(axis=1, keepdims=True)
var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
expected = ((x - mean) / np.sqrt(var + EPS)) * gamma + beta
expected = expected.astype(np.float32)

x.tofile(out / "input_x.bin")
gamma.tofile(out / "input_gamma.bin")
beta.tofile(out / "input_beta.bin")
expected.tofile(out / "expected.bin")

(out / "meta.json").write_text(json.dumps({
    "rows": ROWS,
    "cols": COLS,
    "eps": EPS,
    "dtype": "float32"
}, indent=2))