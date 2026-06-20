import argparse
import csv
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--kind", default=None)
    args = parser.parse_args()

    stage_sum = defaultdict(float)
    stage_count = defaultdict(int)
    total = 0.0

    with open(args.csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if int(row["batch_size"]) != args.batch_size:
                continue

            if args.kind is not None and row["kind"] != args.kind:
                continue

            stage = row["stage"]
            ms = float(row["ms"])

            stage_sum[stage] += ms
            stage_count[stage] += 1
            total += ms

    print("stage,total_ms,count,avg_ms,pct")

    for stage, ms in sorted(stage_sum.items(), key=lambda x: -x[1]):
        count = stage_count[stage]
        avg = ms / count if count else 0.0
        pct = ms / total * 100.0 if total > 0 else 0.0

        print(
            f"{stage},"
            f"{ms:.4f},"
            f"{count},"
            f"{avg:.4f},"
            f"{pct:.2f}"
        )


if __name__ == "__main__":
    main()
