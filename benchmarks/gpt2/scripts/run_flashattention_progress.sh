#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/flashattention_progress}"

WARMUP_RUNS="${WARMUP_RUNS:-3}"
MEASURE_RUNS="${MEASURE_RUNS:-10}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
TOTAL_BLOCKS="${TOTAL_BLOCKS:-8192}"

PROMPTS=("${PROMPTS[@]:-128 256 512 1024}")

mkdir -p "${OUT_DIR}"

COMBINED_CSV="${OUT_DIR}/flashattention_progress_raw.csv"
SUMMARY_CSV="${OUT_DIR}/flashattention_progress_summary.csv"

rm -f "${COMBINED_CSV}" "${SUMMARY_CSV}"

echo "[info] ROOT_DIR=${ROOT_DIR}"
echo "[info] OUT_DIR=${OUT_DIR}"
echo "[info] WARMUP_RUNS=${WARMUP_RUNS}"
echo "[info] MEASURE_RUNS=${MEASURE_RUNS}"
echo "[info] BLOCK_SIZE=${BLOCK_SIZE}"
echo "[info] TOTAL_BLOCKS=${TOTAL_BLOCKS}"
echo "[info] PROMPTS=${PROMPTS[*]}"

# name:use_flash:Br:Bc
VARIANTS=(
  "naive:0:0:0"
  "flash_b4_n32:1:4:32"
  "flash_b4_n64:1:4:64"
  "flash_b8_n32:1:8:32"
  "flash_b8_n64:1:8:64"
  "flash_b16_n32:1:16:32"
)

first_header=1

for variant in "${VARIANTS[@]}"; do
    IFS=":" read -r NAME USE_FLASH BR BC <<< "${variant}"

    BUILD_DIR="${ROOT_DIR}/build_${NAME}"
    LOG_FILE="${OUT_DIR}/${NAME}.log"
    CSV_FILE="${OUT_DIR}/${NAME}.csv"

    echo
    echo "============================================================"
    echo "[build] ${NAME}"
    echo "============================================================"

    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_FLASH_ATTENTION_PREFILL="${USE_FLASH}" \
        -DFA1_BLOCK_M="${BR}" \
        -DFA1_BLOCK_N="${BC}"

    cmake --build "${BUILD_DIR}" -j --target bench_gpt2_wmma_prefill_sweep

    echo
    echo "================================================------------"
    echo "[run] ${NAME}"
    echo "================================================------------"

    "${BUILD_DIR}/bench_gpt2_wmma_prefill_sweep" \
        "${WARMUP_RUNS}" \
        "${MEASURE_RUNS}" \
        "${BLOCK_SIZE}" \
        "${TOTAL_BLOCKS}" \
        "${PROMPTS[@]}" | tee "${LOG_FILE}"

    # CSV header + numeric/string rows만 추출
    awk '
      /^attention_impl,/ { print }
      /^(naive|flashattention1),/ { print }
    ' "${LOG_FILE}" > "${CSV_FILE}"

    if [[ "${first_header}" -eq 1 ]]; then
        cat "${CSV_FILE}" >> "${COMBINED_CSV}"
        first_header=0
    else
        tail -n +2 "${CSV_FILE}" >> "${COMBINED_CSV}"
    fi
done

python3 - <<PY
import csv
from pathlib import Path

combined = Path("${COMBINED_CSV}")
summary = Path("${SUMMARY_CSV}")

rows = []
with combined.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        row["prompt_len"] = int(row["prompt_len"])
        row["use_flash"] = int(row["use_flash"])
        row["fa1_block_m"] = int(row["fa1_block_m"])
        row["fa1_block_n"] = int(row["fa1_block_n"])
        row["gpu_ms_mean"] = float(row["gpu_ms_mean"])
        row["wall_ms_mean"] = float(row["wall_ms_mean"])
        row["prefill_tok_per_s_gpu"] = float(row["prefill_tok_per_s_gpu"])
        row["prefill_tok_per_s_wall"] = float(row["prefill_tok_per_s_wall"])
        rows.append(row)

baseline_by_prompt = {}
for r in rows:
    if r["attention_impl"] == "naive":
        baseline_by_prompt[r["prompt_len"]] = r

fieldnames = [
    "variant",
    "attention_impl",
    "use_flash",
    "fa1_block_m",
    "fa1_block_n",
    "prompt_len",
    "gpu_ms_mean",
    "wall_ms_mean",
    "prefill_tok_per_s_gpu",
    "prefill_tok_per_s_wall",
    "speedup_vs_naive_gpu_ms",
    "speedup_vs_naive_wall_ms",
]

out_rows = []
for r in rows:
    p = r["prompt_len"]
    base = baseline_by_prompt.get(p)

    if base is None:
        speedup_gpu = 0.0
        speedup_wall = 0.0
    else:
        speedup_gpu = base["gpu_ms_mean"] / r["gpu_ms_mean"] if r["gpu_ms_mean"] > 0 else 0.0
        speedup_wall = base["wall_ms_mean"] / r["wall_ms_mean"] if r["wall_ms_mean"] > 0 else 0.0

    if r["attention_impl"] == "naive":
        variant = "naive"
    else:
        variant = f"flash_b{r['fa1_block_m']}_n{r['fa1_block_n']}"

    out_rows.append({
        "variant": variant,
        "attention_impl": r["attention_impl"],
        "use_flash": r["use_flash"],
        "fa1_block_m": r["fa1_block_m"],
        "fa1_block_n": r["fa1_block_n"],
        "prompt_len": p,
        "gpu_ms_mean": f"{r['gpu_ms_mean']:.6f}",
        "wall_ms_mean": f"{r['wall_ms_mean']:.6f}",
        "prefill_tok_per_s_gpu": f"{r['prefill_tok_per_s_gpu']:.6f}",
        "prefill_tok_per_s_wall": f"{r['prefill_tok_per_s_wall']:.6f}",
        "speedup_vs_naive_gpu_ms": f"{speedup_gpu:.6f}",
        "speedup_vs_naive_wall_ms": f"{speedup_wall:.6f}",
    })

with summary.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)

print(f"[saved] {combined}")
print(f"[saved] {summary}")

print()
print("================================================================================")
print("FlashAttention progress summary")
print("================================================================================")

for r in out_rows:
    print(
        f"{r['variant']:16s} "
        f"prompt={r['prompt_len']:>4d} "
        f"gpu_ms={float(r['gpu_ms_mean']):>10.4f} "
        f"tok/s={float(r['prefill_tok_per_s_gpu']):>10.2f} "
        f"speedup={float(r['speedup_vs_naive_gpu_ms']):>7.3f}x"
    )
PY

echo
echo "[done]"
echo "raw csv     : ${COMBINED_CSV}"
echo "summary csv : ${SUMMARY_CSV}"
