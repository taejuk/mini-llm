import argparse
import numpy as np


def compare(name, hf_path, mini_path, rtol=1e-3, atol=1e-3):
    hf = np.load(hf_path)
    mini = np.load(mini_path)

    print(f"\n[{name}]")
    print("hf shape  :", hf.shape)
    print("mini shape:", mini.shape)

    if hf.shape != mini.shape:
        print("❌ shape mismatch")
        return False

    diff = np.abs(hf - mini)

    max_abs = diff.max()
    mean_abs = diff.mean()

    denom = np.maximum(np.abs(hf), 1e-8)
    rel = diff / denom

    max_rel = rel.max()
    mean_rel = rel.mean()

    ok = np.allclose(hf, mini, rtol=rtol, atol=atol)

    print("max_abs :", max_abs)
    print("mean_abs:", mean_abs)
    print("max_rel :", max_rel)
    print("mean_rel:", mean_rel)
    print("result  :", "✅ PASS" if ok else "❌ FAIL")

    if not ok:
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print("worst index:", idx)
        print("hf value   :", hf[idx])
        print("mini value :", mini[idx])

    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--hf", required=True)
    parser.add_argument("--mini", required=True)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    compare(args.name, args.hf, args.mini, args.rtol, args.atol)


if __name__ == "__main__":
    main()