import argparse
import csv


def useful_lines(path):
    with open(path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("batch_size,") or line[0].isdigit():
                yield line


def load_vllm(path):
    rows = {}

    reader = csv.DictReader(useful_lines(path))

    for r in reader:
        batch = int(r["batch_size"])
        rows[batch] = {
            "batch_size": batch,
            "prompt_len": int(r["prompt_len"]),
            "max_new_tokens": int(r["max_new_tokens"]),
            "avg_total_ms": float(r["avg_total_ms"]),
            "tokens_per_sec": float(r["tokens_per_sec"]),
        }

    return rows


def load_minillm(path, mode="batched"):
    rows = {}

    reader = csv.DictReader(useful_lines(path))

    for r in reader:
        if r.get("mode") != mode:
            continue

        batch = int(r["batch_size"])
        rows[batch] = {
            "batch_size": batch,
            "prompt_len": int(r["prompt_len"]),
            "max_new_tokens": int(r["max_new_tokens"]),
            "mode": r["mode"],
            "avg_total_ms": float(r["avg_total_ms"]),
            "tokens_per_sec": float(r["tokens_per_sec"]),
        }

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", required=True)
    parser.add_argument("--mini", required=True)
    parser.add_argument("--mini-mode", default="batched")
    args = parser.parse_args()

    vllm = load_vllm(args.vllm)
    mini = load_minillm(args.mini, args.mini_mode)

    common_batches = sorted(set(vllm.keys()) & set(mini.keys()))

    print(
        "batch_size,"
        "prompt_len,"
        "max_new_tokens,"
        "mini_ms,"
        "vllm_ms,"
        "mini_tok_s,"
        "vllm_tok_s,"
        "vllm_over_mini_tok_s,"
        "mini_over_vllm_latency"
    )

    for b in common_batches:
        m = mini[b]
        v = vllm[b]

        vllm_over_mini_tok_s = (
            v["tokens_per_sec"] / m["tokens_per_sec"]
            if m["tokens_per_sec"] > 0
            else 0.0
        )

        mini_over_vllm_latency = (
            m["avg_total_ms"] / v["avg_total_ms"]
            if v["avg_total_ms"] > 0
            else 0.0
        )

        print(
            f"{b},"
            f"{m['prompt_len']},"
            f"{m['max_new_tokens']},"
            f"{m['avg_total_ms']:.4f},"
            f"{v['avg_total_ms']:.4f},"
            f"{m['tokens_per_sec']:.4f},"
            f"{v['tokens_per_sec']:.4f},"
            f"{vllm_over_mini_tok_s:.4f},"
            f"{mini_over_vllm_latency:.4f}"
        )


if __name__ == "__main__":
    main()
