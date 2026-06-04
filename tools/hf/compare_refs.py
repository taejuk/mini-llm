#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np


DTYPE_MAP = {
    "float32": np.float32,
    "int32": np.int32,
    "int64": np.int64,
}


# hf_ref 파일 이름 -> mini_ref tensor 이름
# 필요할 때 여기에 계속 추가하면 됨.
DEFAULT_NAME_MAP: Dict[str, str] = {
    "embedding_out": "wmma_embedding_out",
    "hidden_state_1": "wmma_hidden_state_1",
    "final_ln_out": "wmma_final_ln_out",
    "logits": "wmma_logits",
}


def load_hf_tensor(hf_dir: Path, name: str) -> np.ndarray:
    path = hf_dir / f"{name}.npy"

    if not path.exists():
        raise FileNotFoundError(f"HF tensor not found: {path}")

    return np.load(path)


def load_mini_tensor(mini_dir: Path, name: str) -> np.ndarray:
    shape_path = mini_dir / f"{name}.shape"
    bin_path = mini_dir / f"{name}.bin"

    if not shape_path.exists():
        raise FileNotFoundError(f"mini shape file not found: {shape_path}")

    if not bin_path.exists():
        raise FileNotFoundError(f"mini bin file not found: {bin_path}")

    with open(shape_path, "r") as f:
        dtype_name = f.readline().strip()
        shape_line = f.readline().strip()

    if dtype_name not in DTYPE_MAP:
        raise ValueError(f"unknown dtype '{dtype_name}' in {shape_path}")

    shape = tuple(map(int, shape_line.split()))
    dtype = DTYPE_MAP[dtype_name]

    arr = np.fromfile(bin_path, dtype=dtype)

    expected = int(np.prod(shape))
    if arr.size != expected:
        raise RuntimeError(
            f"size mismatch for {name}: "
            f"file has {arr.size} elements, shape expects {expected}, shape={shape}"
        )

    return arr.reshape(shape)


def compare_tensors(
    hf: np.ndarray,
    mini: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> Tuple[bool, Dict[str, object]]:
    if hf.shape != mini.shape:
        return False, {
            "reason": "shape_mismatch",
            "hf_shape": hf.shape,
            "mini_shape": mini.shape,
        }

    # dtype 차이는 허용. 비교는 float64로 올려서 안정적으로 계산.
    hf_f = hf.astype(np.float64)
    mini_f = mini.astype(np.float64)

    diff = np.abs(hf_f - mini_f)

    max_abs = float(diff.max()) if diff.size > 0 else 0.0
    mean_abs = float(diff.mean()) if diff.size > 0 else 0.0

    denom = np.maximum(np.abs(hf_f), 1e-12)
    rel = diff / denom

    max_rel = float(rel.max()) if rel.size > 0 else 0.0
    mean_rel = float(rel.mean()) if rel.size > 0 else 0.0

    ok = bool(np.allclose(hf_f, mini_f, atol=atol, rtol=rtol))

    worst_idx = np.unravel_index(np.argmax(diff), diff.shape) if diff.size > 0 else None

    result = {
        "reason": "ok" if ok else "value_mismatch",
        "hf_shape": hf.shape,
        "mini_shape": mini.shape,
        "hf_dtype": str(hf.dtype),
        "mini_dtype": str(mini.dtype),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
        "mean_rel": mean_rel,
        "worst_idx": worst_idx,
    }

    if worst_idx is not None:
        result["hf_worst"] = float(hf_f[worst_idx])
        result["mini_worst"] = float(mini_f[worst_idx])

    return ok, result


def print_result(hf_name: str, mini_name: str, ok: bool, info: Dict[str, object]) -> None:
    status = "✅ PASS" if ok else "❌ FAIL"

    print(f"\n[{hf_name}  <->  {mini_name}]")
    print(f"result     : {status}")

    if info["reason"] == "shape_mismatch":
        print(f"reason     : shape mismatch")
        print(f"hf shape   : {info['hf_shape']}")
        print(f"mini shape : {info['mini_shape']}")
        return

    print(f"hf shape   : {info['hf_shape']}")
    print(f"mini shape : {info['mini_shape']}")
    print(f"hf dtype   : {info['hf_dtype']}")
    print(f"mini dtype : {info['mini_dtype']}")
    print(f"max abs    : {info['max_abs']:.8e}")
    print(f"mean abs   : {info['mean_abs']:.8e}")
    print(f"max rel    : {info['max_rel']:.8e}")
    print(f"mean rel   : {info['mean_rel']:.8e}")

    if not ok:
        print(f"worst idx  : {info['worst_idx']}")
        print(f"hf value   : {info['hf_worst']:.8e}")
        print(f"mini value : {info['mini_worst']:.8e}")


def parse_name_map(items: Optional[list[str]]) -> Dict[str, str]:
    name_map = dict(DEFAULT_NAME_MAP)

    if not items:
        return name_map

    for item in items:
        if "=" not in item:
            raise ValueError(
                f"invalid --map item: {item}. "
                f"Expected format: hf_name=mini_name"
            )

        hf_name, mini_name = item.split("=", 1)
        hf_name = hf_name.strip()
        mini_name = mini_name.strip()

        if not hf_name or not mini_name:
            raise ValueError(f"invalid --map item: {item}")

        name_map[hf_name] = mini_name

    return name_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Hugging Face reference tensors with mini-llm dumped tensors."
    )

    parser.add_argument("--hf-dir", default="hf_ref", help="directory containing HF .npy files")
    parser.add_argument("--mini-dir", default="mini_ref", help="directory containing mini .bin/.shape files")

    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help=(
            "HF tensor names to compare. "
            "Example: --names embedding_out logits"
        ),
    )

    parser.add_argument(
        "--map",
        nargs="*",
        default=None,
        help=(
            "Additional name mappings. "
            "Format: hf_name=mini_name. "
            "Example: --map embedding_out=wmma_embedding_out"
        ),
    )

    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)

    args = parser.parse_args()

    hf_dir = Path(args.hf_dir)
    mini_dir = Path(args.mini_dir)

    name_map = parse_name_map(args.map)

    if args.names:
        hf_names = args.names
    else:
        hf_names = list(name_map.keys())

    total = 0
    passed = 0

    print(f"hf_dir   : {hf_dir}")
    print(f"mini_dir : {mini_dir}")
    print(f"atol     : {args.atol}")
    print(f"rtol     : {args.rtol}")

    for hf_name in hf_names:
        mini_name = name_map.get(hf_name, hf_name)

        total += 1

        try:
            hf = load_hf_tensor(hf_dir, hf_name)
            mini = load_mini_tensor(mini_dir, mini_name)

            ok, info = compare_tensors(
                hf,
                mini,
                atol=args.atol,
                rtol=args.rtol,
            )

            print_result(hf_name, mini_name, ok, info)

            if ok:
                passed += 1

        except Exception as e:
            print(f"\n[{hf_name}  <->  {mini_name}]")
            print("result     : ❌ ERROR")
            print(f"error      : {e}")

    print("\n========== Summary ==========")
    print(f"passed: {passed}/{total}")

    if passed == total:
        print("✅ ALL PASS")
    else:
        print("❌ SOME TESTS FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()