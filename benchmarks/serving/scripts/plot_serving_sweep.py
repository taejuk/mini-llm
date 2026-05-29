#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


N_LAYERS = 12


def blocks_per_request(prompt_len, output_len, block_size):
    return math.ceil((prompt_len + output_len) / block_size) * N_LAYERS


def total_blocks_needed(num_requests, prompt_len, output_len, block_size):
    return num_requests * blocks_per_request(prompt_len, output_len, block_size)


def load_csv_if_exists(path):
    path = Path(path)

    if not path.exists():
        print(f"[skip] missing: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    return df


def plot_metric_vs_requests(df, out_dir, metric, ylabel):
    if df.empty or metric not in df.columns:
        return

    group_cols = ["prompt_len", "output_len", "block_size"]

    for keys, sub in df.groupby(group_cols):
        prompt_len, output_len, block_size = keys

        plt.figure(figsize=(10, 6))

        for (mode, fit_case), g in sub.groupby(["mode", "fit_case"]):
            g = g.sort_values("num_requests")

            plt.plot(
                g["num_requests"],
                g[metric],
                marker="o",
                label=f"{mode}-{fit_case}",
            )

        plt.title(
            f"{metric} vs num_requests "
            f"(prompt={prompt_len}, output={output_len}, block_size={block_size})"
        )
        plt.xlabel("num_requests")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        fname = (
            f"{metric}_vs_requests"
            f"_p{prompt_len}"
            f"_o{output_len}"
            f"_bs{block_size}.png"
        )

        plt.savefig(out_dir / fname, dpi=150)
        plt.close()


def plot_metric_vs_total_blocks(df, out_dir, metric, ylabel):
    if df.empty or metric not in df.columns:
        return

    group_cols = ["num_requests", "prompt_len", "output_len", "block_size"]

    for keys, sub in df.groupby(group_cols):
        num_requests, prompt_len, output_len, block_size = keys

        plt.figure(figsize=(10, 6))

        for mode, g in sub.groupby("mode"):
            g = g.sort_values("total_blocks")

            plt.plot(
                g["total_blocks"],
                g[metric],
                marker="o",
                label=mode,
            )

        needed = total_blocks_needed(
            int(num_requests),
            int(prompt_len),
            int(output_len),
            int(block_size),
        )

        plt.axvline(
            needed,
            linestyle="--",
            label=f"all-fit boundary={needed}",
        )

        plt.title(
            f"{metric} vs total_blocks "
            f"(req={num_requests}, prompt={prompt_len}, output={output_len}, block_size={block_size})"
        )
        plt.xlabel("total_blocks")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        fname = (
            f"{metric}_vs_total_blocks"
            f"_req{num_requests}"
            f"_p{prompt_len}"
            f"_o{output_len}"
            f"_bs{block_size}.png"
        )

        plt.savefig(out_dir / fname, dpi=150)
        plt.close()


def plot_decode_batch_speedup(df, out_dir):
    if df.empty:
        return

    required = {
        "mode",
        "fit_case",
        "num_requests",
        "prompt_len",
        "output_len",
        "block_size",
        "out_tok_per_s",
        "tpot_mean",
    }

    if not required.issubset(set(df.columns)):
        return

    group_cols = ["fit_case", "prompt_len", "output_len", "block_size"]

    for keys, sub in df.groupby(group_cols):
        fit_case, prompt_len, output_len, block_size = keys

        pivot = sub.pivot_table(
            index="num_requests",
            columns="mode",
            values="out_tok_per_s",
            aggfunc="mean",
        )

        if "batch" not in pivot.columns or "nobatch" not in pivot.columns:
            continue

        speedup = pivot["batch"] / pivot["nobatch"]

        plt.figure(figsize=(10, 6))
        plt.plot(speedup.index, speedup.values, marker="o")
        plt.title(
            f"decode batching speedup "
            f"(fit_case={fit_case}, prompt={prompt_len}, output={output_len}, block_size={block_size})"
        )
        plt.xlabel("num_requests")
        plt.ylabel("out_tok_per_s speedup: batch / nobatch")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fname = (
            f"decode_batch_speedup"
            f"_{fit_case}"
            f"_p{prompt_len}"
            f"_o{output_len}"
            f"_bs{block_size}.png"
        )

        plt.savefig(out_dir / fname, dpi=150)
        plt.close()


def print_summary(df, title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    if df.empty:
        print("empty")
        return

    cols = [
        "mode",
        "fit_case",
        "num_requests",
        "prompt_len",
        "output_len",
        "block_size",
        "total_blocks",
        "completed",
        "out_tok_per_s",
        "ttft_mean",
        "tpot_mean",
        "e2e_mean",
        "peak_used_blocks",
    ]

    cols = [c for c in cols if c in df.columns]

    print(
        df[cols]
        .sort_values(["prompt_len", "output_len", "num_requests", "mode", "fit_case"])
        .to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv-dir",
        default="benchmarks/serving/results/serving_sweep",
        help="Directory containing serving_sweep_results.csv and block_capacity_sweep_results.csv",
    )

    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Output directory for plots. Default: <csv-dir>/plots",
    )

    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    plot_dir = Path(args.plot_dir) if args.plot_dir else csv_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    serving_csv = csv_dir / "serving_sweep_results.csv"
    block_csv = csv_dir / "block_capacity_sweep_results.csv"

    serving_df = load_csv_if_exists(serving_csv)
    block_df = load_csv_if_exists(block_csv)

    print_summary(serving_df, "Main serving sweep")
    print_summary(block_df, "Block capacity sweep")

    metrics = [
        ("out_tok_per_s", "output tokens / sec"),
        ("total_tok_per_s", "total tokens / sec"),
        ("req_per_s", "requests / sec"),
        ("ttft_mean", "TTFT mean (ms)"),
        ("ttft_p50", "TTFT p50 (ms)"),
        ("ttft_p95", "TTFT p95 (ms)"),
        ("tpot_mean", "TPOT mean (ms)"),
        ("tpot_p50", "TPOT p50 (ms)"),
        ("tpot_p95", "TPOT p95 (ms)"),
        ("e2e_mean", "E2E mean latency (ms)"),
        ("e2e_p50", "E2E p50 latency (ms)"),
        ("e2e_p95", "E2E p95 latency (ms)"),
        ("peak_used_blocks", "peak used KV blocks"),
    ]

    for metric, ylabel in metrics:
        plot_metric_vs_requests(serving_df, plot_dir, metric, ylabel)
        plot_metric_vs_total_blocks(block_df, plot_dir, metric, ylabel)

    plot_decode_batch_speedup(serving_df, plot_dir)

    print()
    print(f"Saved plots to: {plot_dir}")


if __name__ == "__main__":
    main()
