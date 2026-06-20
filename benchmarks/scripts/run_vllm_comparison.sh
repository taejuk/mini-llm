#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RESULT_DIR="benchmarks/results"
mkdir -p "$RESULT_DIR"

echo "Saving results to $RESULT_DIR"

echo "[mini-llm] prompt_len=5 max_new_tokens=8"
./build/bench_batch_vs_single_wall 5 8 5 1 \
  | grep -E '^(batch_size|[0-9]+,)' \
  | tee "$RESULT_DIR/mini_5_8_wall.csv"

echo "[vLLM FP32] prompt_len=5 max_new_tokens=8"
python3 benchmarks/vllm/bench_vllm_token.py 5 8 5 1 \
  2> "$RESULT_DIR/vllm_float_5_8.log" \
  | grep -E '^(batch_size|[0-9]+,)' \
  | tee "$RESULT_DIR/vllm_float_5_8.csv"

echo "[compare] prompt_len=5 max_new_tokens=8"
python3 benchmarks/compare/compare_vllm_minillm.py \
  --mini "$RESULT_DIR/mini_5_8_wall.csv" \
  --vllm "$RESULT_DIR/vllm_float_5_8.csv" \
  | tee "$RESULT_DIR/compare_5_8.csv"


echo "[mini-llm] prompt_len=256 max_new_tokens=1"
./build/bench_batch_vs_single_wall 256 1 5 1 \
  | grep -E '^(batch_size|[0-9]+,)' \
  | tee "$RESULT_DIR/mini_256_1_wall.csv"

echo "[vLLM FP32] prompt_len=256 max_new_tokens=1"
python3 benchmarks/vllm/bench_vllm_token.py 256 1 5 1 \
  2> "$RESULT_DIR/vllm_float_256_1.log" \
  | grep -E '^(batch_size|[0-9]+,)' \
  | tee "$RESULT_DIR/vllm_float_256_1.csv"

echo "[compare] prompt_len=256 max_new_tokens=1"
python3 benchmarks/compare/compare_vllm_minillm.py \
  --mini "$RESULT_DIR/mini_256_1_wall.csv" \
  --vllm "$RESULT_DIR/vllm_float_256_1.csv" \
  | tee "$RESULT_DIR/compare_256_1.csv"


echo "[mini-llm] prompt_len=256 max_new_tokens=8"
./build/bench_batch_vs_single_wall 256 8 10 1 \
  | grep -E '^(batch_size|[0-9]+,)' \
  | tee "$RESULT_DIR/mini_256_8_wall.csv"

echo "[vLLM FP32] prompt_len=256 max_new_tokens=8"
python3 benchmarks/vllm/bench_vllm_token.py 256 8 10 1 \
  2> "$RESULT_DIR/vllm_float_256_8.log" \
  | grep -E '^(batch_size|[0-9]+,)' \
  | tee "$RESULT_DIR/vllm_float_256_8.csv"

echo "[compare] prompt_len=256 max_new_tokens=8"
python3 benchmarks/compare/compare_vllm_minillm.py \
  --mini "$RESULT_DIR/mini_256_8_wall.csv" \
  --vllm "$RESULT_DIR/vllm_float_256_8.csv" \
  | tee "$RESULT_DIR/compare_256_8.csv"

echo "Done. Results saved to $RESULT_DIR"
