# residual_add에 대한 정답 케이스 생성
import numpy as np
from pathlib import Path
import json

out = Path(__file__).parent
N = 1027

x = np.arange(N, dtype=np.float32) * 0.01
y = 1.0 + np.arange(N, dtype=np.float32) * 0.001
expected = x + y
#residual_add 연산
x.tofile(out / "input_x.bin")
y.tofile(out / "input_y.bin")
expected.tofile(out / "expected.bin")

(out / "meta.json").write_text(json.dumps({
    "N": N,
    "dtype": "float32"
}, indent=2))