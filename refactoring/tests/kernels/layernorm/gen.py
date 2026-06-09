import numpy as np
from pathlib import Path
import json

def layernorm(x, gamma, beta, eps):
    row_mean = np.mean(x, axis=1)
    # 각 row의 평균을 뺸다.

    # 분산을 구한다.

    # gamma와 beta를 구한다.
out = Path(__file__).parent
N = 1027
M = 4096

x = np.random.rand(N, M)
gamma = np.random.rand(N, M)
beta = np.random.rand(N, M)
eps = 1e-9

expected = layernorm(x, gamma, beta, eps)

x.tofile(out / "input_x.bin")
gamma.tofile(out / "input_gamma.bin")
beta.tofile(out / "input_beta.bin")
expected.tofile(out / "expected.bin")


(out / "meta.json").write_text(json.dumps({
    "N": N,
    "M": M
}))
