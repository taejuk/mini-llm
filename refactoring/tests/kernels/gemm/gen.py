import json
from pathlib import Path

import numpy as np

out = Path(__file__).parent

# tile boundary를 테스트하려고 일부러 64의 배수가 아닌 shape 사용
ROWS = 67
COLS = 79
K = 37

ALPHA = 0.5
BETA = 0.25

rng = np.random.default_rng(1)

A = rng.normal(0.0, 1.0, size=(ROWS, K)).astype(np.float32)
B = rng.normal(0.0, 1.0, size=(K, COLS)).astype(np.float32)
C = rng.normal(0.0, 1.0, size=(ROWS, COLS)).astype(np.float32)

expected = (BETA * C + ALPHA * (A @ B)).astype(np.float32)

A.tofile(out / "input_x.bin")
B.tofile(out / "input_y.bin")
C.tofile(out / "input_z.bin")
expected.tofile(out / "expected.bin")

(out / "meta.json").write_text(json.dumps({
    "rows": ROWS,
    "cols": COLS,
    "K": K,
    "alpha": ALPHA,
    "beta": BETA,
    "dtype": "float32"
}, indent=2))