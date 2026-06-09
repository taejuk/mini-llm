import numpy as np
from pathlib import Path
import json

out = Path(__file__).parent

N = 1027
M = 4048
K = 486
alpha = 0.5
beta = 0.5

x = np.random.rand(N, K)
y = np.random.rand(K, M)
z = np.random.rand(N, M)
# 데이터를 써야 한다.
x.tofile(out / "input_x.bin")
y.tofile(out / "input_y.bin")
z.tofile(out / "input_z.bin")

z = beta * z + alpha * x @ y

z.tofile(out / "expected.bin")

(out / "meta.json").write_text(json.dumps({
    "N" : N,
    "M": M,
    "K": K,
    "dtype": "float32"
}, indent=2))