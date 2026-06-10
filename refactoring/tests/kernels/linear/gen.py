import json
import numpy as np
from pathlib import Path

out = Path(__file__).parent
rng = np.random.default_rng(0)

M = 5
K = 768
N = 2304

x = rng.normal(0, 0.02, size=(M, K)).astype(np.float32)
w = rng.normal(0, 0.02, size=(K, N)).astype(np.float32)
b = rng.normal(0, 0.02, size=(N,)).astype(np.float32)

expected = x @ w + b

x.tofile(out / "input_x.bin")
w.tofile(out / "weight.bin")
b.tofile(out / "bias.bin")
expected.astype(np.float32).tofile(out / "expected.bin")

(out / "meta.json").write_text(json.dumps({
    "M": M,
    "K": K,
    "N": N,
    "dtype": "float32"
}, indent=2))