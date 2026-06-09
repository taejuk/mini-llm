import numpy as np
from pathlib import Path
import json

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715* np.power(x,3))));

out = Path(__file__).parent
N = 1027

x = np.arange(N, dtype=np.float32) * 0.01
expected = gelu(x)

x.tofile(out / "input_x.bin")
expected.tofile(out / "expected.bin")

(out / "meta.json").write_text(json.dumps({
    "N": N,
    "dtype": "float32"
}, indent=2))