#!/usr/bin/env python3

import argparse
import csv
import math
import subprocess
from pathlib import Path


N_LAYERS = 12


def blocks_per_request(prompt_len, output_len, block_size):
    return math.ceil((prompt_len + output_len) / block_size) * N_LAYERS


def total_blocks_needed(num_requests, prompt_len, output_len, block_size):
    return num_requests * blocks_per_request(prompt_len, output_len, block_size)


def choose_total_blocks(num_requests, prompt_len, output_len, block_size, case):
    per_req = blocks_per_request(prompt_len, output_len, block_size)
    needed = per_req * num_requests

    if case == "fit":
        return int(math.ceil(needed * 1.10))

    if case == "not_fit":
        return max(per_req, int(math.floor(needed * 0.50)))

    raise ValueError(f"unknown case: {case}")


def parse_benchmark_stdout(stdout):
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("num_requests,prompt_len,output_len"):
            header_idx = i

    if header_idx is None:
        raise RuntimeError("Could not find benchmark CSV header in stdout")

    if header_idx + 1 >= len(lines):
        raise RuntimeError("Could not find benchmark CSV row in stdout")

    header = lines[header_idx].split(",")
    row = lines[header_idx + 1].split(",")

    if len(header) != len(row):
        raise RuntimeError(
            f"CSV length mismatch: header={len(header)} row={len(row)}"
        )

    result = {}
    for k, v in zip(header, row):
        try:
            if "." in v:
                result[k] = float(v)
            else:
                result[k] = int(v)
        except ValueError:
            result[k] = v

    return result


def write_rows_csv(path, rows):
    if not rows:
        return

    fieldnames = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def run_one(
    exe_path,
    num_requests,
    prompt_len,
    output_len,
    block_size,
    total_blocks,
    timeout_sec,
    raw_log_dir,
    tag,
):
    cmd = [
        str(exe_path),
        str(num_requests),
        str(prompt_len),
        str(output_len),
        str(block_size),
        str(total_blocks),
        "0",
    ]

    raw_log_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_sec,
    )

    log_prefix = (
        f"{tag}"
        f"_req{num_requests}"
        f"_p{prompt_len}"
        f"_o{output_len}"
        f"_bs{block_size}"
        f"_blocks{total_blocks}"
    )

    (raw_log_dir / f"{log_prefix}.stdout.txt").write_text(proc.stdout)
    (raw_log_dir / f"{log_prefix}.stderr.txt").write_text(proc.stderr)

    if proc.returncode != 0:
        return {
            "status": "failed",
            "error": proc.stderr[-1000:],
            "cmd": " ".join(cmd),
        }

    try:
        row = parse_benchmark_stdout(proc.stdout)
        row["status"] = "ok"
        row["cmd"] = " ".join(cmd)
        return row
    except Exception as e:
        return {
            "status": "parse_failed",
            "error": str(e),
            "cmd": " ".join(cmd),
        }


def add_metadata(
    row,
    mode,
    fit_case,
    num_requests,
    prompt_len,
    output_len,
    block_size,
    total_blocks,
):
    per_req = blocks_per_request(prompt_len, output_len, block_size)
    needed = total_blocks_needed(num_requests, prompt_len, output_len, block_size)

    meta = {
        "mode": mode,
        "fit_case": fit_case,
        "num_requests": num_requests,
        "prompt_len": prompt_len,
        "output_len": output_len,
        "block_size": block_size,
        "total_blocks": total_blocks,
        "blocks_per_request_calc": per_req,
        "total_blocks_needed_calc": needed,
        "all_requests_fit_expected": total_blocks >= needed,
    }

    meta.update(row)
    return meta


def run_main_sweep(args):
    build_dir = Path(args.build_dir)
    out_dir = Path(args.out_dir)
    raw_log_dir = out_dir / "raw_logs"

    executables = {
        "batch": build_dir / "bench_offline_gpt2_wmma",
        "nobatch": build_dir / "bench_offline_gpt2_wmma_decode_nobatch",
    }

    rows = []

    for mode in args.modes:
        exe_path = executables[mode]

        if not exe_path.exists():
            raise FileNotFoundError(f"Executable not found: {exe_path}")

        for prompt_len in args.prompt_lens:
            for output_len in args.output_lens:
                for block_size in args.block_sizes:
                    for num_requests in args.num_requests:
                        for fit_case in args.fit_cases:
                            total_blocks = choose_total_blocks(
                                num_requests,
                                prompt_len,
                                output_len,
                                block_size,
                                fit_case,
                            )

                            print(
                                "[run]",
                                f"mode={mode}",
                                f"case={fit_case}",
                                f"req={num_requests}",
                                f"prompt={prompt_len}",
                                f"output={output_len}",
                                f"block_size={block_size}",
                                f"total_blocks={total_blocks}",
                                flush=True,
                            )

                            try:
                                row = run_one(
                                    exe_path=exe_path,
                                    num_requests=num_requests,
                                    prompt_len=prompt_len,
                                    output_len=output_len,
                                    block_size=block_size,
                                    total_blocks=total_blocks,
                                    timeout_sec=args.timeout_sec,
                                    raw_log_dir=raw_log_dir,
                                    tag=f"{mode}_{fit_case}",
                                )
                            except subprocess.TimeoutExpired as e:
                                row = {
                                    "status": "timeout",
                                    "error": str(e),
                                }

                            row = add_metadata(
                                row=row,
                                mode=mode,
                                fit_case=fit_case,
                                num_requests=num_requests,
                                prompt_len=prompt_len,
                                output_len=output_len,
                                block_size=block_size,
                                total_blocks=total_blocks,
                            )

                            rows.append(row)
                            write_rows_csv(out_dir / "serving_sweep_results.csv", rows)

    return rows


def run_block_capacity_sweep(args):
    build_dir = Path(args.build_dir)
    out_dir = Path(args.out_dir)
    raw_log_dir = out_dir / "raw_logs_block_sweep"

    executables = {
        "batch": build_dir / "bench_offline_gpt2_wmma",
        "nobatch": build_dir / "bench_offline_gpt2_wmma_decode_nobatch",
    }

    rows = []

    num_requests = args.block_sweep_num_requests
    prompt_len = args.block_sweep_prompt_len
    output_len = args.block_sweep_output_len
    block_size = args.block_sweep_block_size

    per_req = blocks_per_request(prompt_len, output_len, block_size)
    needed = total_blocks_needed(num_requests, prompt_len, output_len, block_size)

    for mode in args.modes:
        exe_path = executables[mode]

        if not exe_path.exists():
            raise FileNotFoundError(f"Executable not found: {exe_path}")

        for factor in args.block_factors:
            total_blocks = max(per_req, int(math.ceil(needed * factor)))
            fit_case = "fit" if total_blocks >= needed else "not_fit"

            print(
                "[block-sweep]",
                f"mode={mode}",
                f"factor={factor}",
                f"case={fit_case}",
                f"req={num_requests}",
                f"prompt={prompt_len}",
                f"output={output_len}",
                f"block_size={block_size}",
                f"total_blocks={total_blocks}",
                flush=True,
            )

            try:
                row = run_one(
                    exe_path=exe_path,
                    num_requests=num_requests,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    block_size=block_size,
                    total_blocks=total_blocks,
                    timeout_sec=args.timeout_sec,
                    raw_log_dir=raw_log_dir,
                    tag=f"block_sweep_{mode}_{factor}",
                )
            except subprocess.TimeoutExpired as e:
                row = {
                    "status": "timeout",
                    "error": str(e),
                }

            row = add_metadata(
                row=row,
                mode=mode,
                fit_case=fit_case,
                num_requests=num_requests,
                prompt_len=prompt_len,
                output_len=output_len,
                block_size=block_size,
                total_blocks=total_blocks,
            )

            row["block_factor"] = factor

            rows.append(row)
            write_rows_csv(out_dir / "block_capacity_sweep_results.csv", rows)

    return rows


def print_simple_summary(rows, title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    ok_rows = [r for r in rows if r.get("status") == "ok"]

    if not ok_rows:
        print("No successful runs.")
        return

    keys = [
        "mode",
        "fit_case",
        "num_requests",
        "prompt_len",
        "output_len",
        "total_blocks",
        "completed",
        "out_tok_per_s",
        "ttft_mean",
        "tpot_mean",
        "e2e_mean",
        "peak_used_blocks",
    ]

    print(",".join(keys))
    for r in ok_rows:
        print(",".join(str(r.get(k, "")) for k in keys))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--build-dir", default="./build")
    parser.add_argument("--out-dir", default="benchmarks/serving/results/serving_sweep")

    parser.add_argument(
        "--modes",
        nargs="+",
        default=["batch", "nobatch"],
        choices=["batch", "nobatch"],
    )

    parser.add_argument(
        "--fit-cases",
        nargs="+",
        default=["fit", "not_fit"],
        choices=["fit", "not_fit"],
    )

    parser.add_argument("--num-requests", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--prompt-lens", nargs="+", type=int, default=[32, 128, 512])
    parser.add_argument("--output-lens", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[16])

    parser.add_argument("--timeout-sec", type=int, default=300)

    parser.add_argument("--skip-main-sweep", action="store_true")
    parser.add_argument("--skip-block-sweep", action="store_true")

    parser.add_argument("--block-sweep-num-requests", type=int, default=32)
    parser.add_argument("--block-sweep-prompt-len", type=int, default=128)
    parser.add_argument("--block-sweep-output-len", type=int, default=32)
    parser.add_argument("--block-sweep-block-size", type=int, default=16)

    parser.add_argument(
        "--block-factors",
        nargs="+",
        type=float,
        default=[0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00],
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    main_rows = []
    block_rows = []

    if not args.skip_main_sweep:
        main_rows = run_main_sweep(args)
        print_simple_summary(main_rows, "Main serving sweep")

    if not args.skip_block_sweep:
        block_rows = run_block_capacity_sweep(args)
        print_simple_summary(block_rows, "Block capacity sweep")

    print()
    print(f"Saved CSV results to: {out_dir}")
    print(f"Main sweep CSV:        {out_dir / 'serving_sweep_results.csv'}")
    print(f"Block sweep CSV:       {out_dir / 'block_capacity_sweep_results.csv'}")
    print()
    print("Plotting is separated. Copy the CSV files to a machine with matplotlib and run:")
    print("  python3 benchmarks/serving/scripts/plot_serving_sweep.py --csv-dir <csv_dir>")


if __name__ == "__main__":
    main()
