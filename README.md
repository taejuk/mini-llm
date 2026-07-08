# mini-llm

`mini-llm` is a GPT-2 inference and serving engine implemented from scratch in C++ and CUDA.

The goal of this project is not only to run a GPT-2 style model, but to understand the systems problems behind modern LLM serving: GPU kernels, prefill/decode separation, batching, paged KV cache management, scheduling, CPU/GPU data movement, and asynchronous server runtime design.

This repository is an educational systems project inspired by vLLM, Orca, FlashAttention, and GPU performance engineering papers.

---

## Highlights

- GPT-2 small inference pipeline implemented in C++/CUDA
- Custom CUDA kernels for GEMM, LayerNorm, GELU, residual add, argmax, prefill attention, and paged decode attention
- FlashAttention-style prefill attention using tiled causal attention and online softmax
- Paged KV cache manager inspired by vLLM PagedAttention
- Prefill/decode separated scheduler with deferred queues for KV block shortage
- GPU/CPU KV block allocator with swap-out and buffered swap-in path
- libuv-based asynchronous TCP server runtime
- Correctness tests against NumPy, CPU references, and Hugging Face GPT-2 golden outputs
- Benchmark programs for batching, prefill/decode latency, argmax, GEMM, and KV cache swapping

---

## Key Results

Representative results from my benchmark experiments:

| Optimization                    | Result                                                                              |
| ------------------------------- | ----------------------------------------------------------------------------------- |
| FP32 SGEMM optimization         | Reduced 4096x4096 SGEMM latency from **666.3 ms** to **11.9 ms**                    |
| Batched inference               | Improved throughput by **14.6x** on `prompt_len=5`, `max_new_tokens=8`              |
| cuBLAS backend for prefill GEMM | Improved TTFT by **1.30x** at `prompt_len=256`                                      |
| GPU argmax                      | Improved argmax path by **42.5x** at batch size 64 by removing full-logits D2H copy |
| Decode-heavy workload           | Reached **465 tokens/s**, about **43.5% of vLLM throughput** under the same setup   |
| Buffered KV swap                | Improved KV block swap performance by **1.24x** using buffered swap-in              |

These numbers are intended to show the optimization process and bottleneck analysis rather than claim production-level performance.

---

## Why This Project Exists

LLM serving is different from ordinary batched inference.

- Each request has a different prompt length and generation length.
- Prefill and decode have very different compute/memory characteristics.
- KV cache grows token by token during decode.
- Contiguous KV allocation causes fragmentation and admission problems.
- Small-batch decode has poor GPU parallelism unless kernels and scheduling are designed carefully.
- CPU/GPU synchronization and data movement can dominate latency in seemingly small operations.

`mini-llm` explores these issues by building a minimal GPT-2 serving stack from the bottom up.

---

## Architecture

```text
Client
  |
  v
libuv TCP Server / Event Loop
  |
  v
Request Queue
  |
  v
Scheduler
  |-- waiting_prefill_queue
  |-- prefill_queue
  |-- decode_queue
  |-- deferred_prefill_queue
  |-- deferred_decode_queue
  |-- swap_out_queue
  |-- finish_queue
  |-- cancel_queue
  |
  v
GPT2Model
  |
  v
CUDA Kernels
  |-- GEMM / Linear
  |-- LayerNorm
  |-- GELU
  |-- Residual Add
  |-- FlashAttention-style Prefill
  |-- Paged Decode Attention
  |-- Argmax
  |
  v
Paged KV Cache Pool
  |-- GPU KV blocks
  |-- CPU swap blocks
  |-- Block manager
```

---

## Runtime Flow

```text
New request
  |
  v
waiting_prefill_queue
  |
  |  KV block admission
  v
prefill_queue
  |
  |  model.prefill()
  v
decode_queue
  |
  |  model.decode()
  v
finish_queue
```

When KV cache allocation fails, the scheduler does not immediately fail the request. Instead, it moves the request into a deferred queue and retries it later.

```text
Prefill KV admission failed
  -> deferred_prefill_queue

Decode needs a new KV block but GPU memory is insufficient
  -> deferred_decode_queue
```

If the working set cannot make progress due to GPU KV pressure, the scheduler can swap out a victim request's KV cache to CPU memory and retry swap-in later.

---

## Implemented Components

### 1. GPT-2 Inference Pipeline

The model implementation loads GPT-2 small weights exported from Hugging Face into binary files and runs a GPT-2 style forward path.

Implemented model stages:

- Token and positional embedding
- Transformer block loop
- LayerNorm
- QKV projection
- Causal prefill attention
- Paged decode attention
- Attention output projection + residual
- FFN projection + GELU
- FFN output projection + residual
- Final LayerNorm
- Vocabulary projection
- GPU argmax for greedy decoding

The constants match GPT-2 small:

| Parameter           | Value |
| ------------------- | ----- |
| Layers              | 12    |
| Heads               | 12    |
| Hidden size         | 768   |
| Head size           | 64    |
| FFN size            | 3072  |
| Vocabulary size     | 50257 |
| Max sequence length | 1024  |

---

### 2. Custom CUDA Kernels

| Kernel                       | Key Idea                                               | Used For                                       |
| ---------------------------- | ------------------------------------------------------ | ---------------------------------------------- |
| GEMM / Linear                | Shared-memory tiling, vectorized load, scalar fallback | QKV, projection, FFN, logits                   |
| Decode GEMM                  | Smaller output tiling for small `M`                    | Low-batch decode GEMM                          |
| LayerNorm                    | Row-wise reduction                                     | Transformer normalization                      |
| GELU                         | Elementwise activation                                 | FFN activation                                 |
| Linear + GELU                | Kernel fusion                                          | Reduce launch overhead and intermediate writes |
| Linear + Residual            | Kernel fusion                                          | Attention/FFN residual path                    |
| FlashAttention-style Prefill | Tiling + online softmax + causal mask                  | Prompt attention                               |
| Append KV                    | Block/offset based writes                              | Store K/V into paged KV cache                  |
| Paged Decode Attention       | Block-table lookup                                     | Decode attention over non-contiguous KV blocks |
| Argmax                       | GPU reduction over vocab logits                        | Greedy next-token selection                    |

---

### 3. GEMM Optimization

GEMM dominates Transformer inference because QKV projection, attention output projection, FFN layers, and final vocabulary projection are all GEMM-like operations.

Optimizations explored:

- Shared-memory tiling to reduce global memory traffic
- Register blocking
- Coalesced memory access
- `float4` vectorized loads
- Scalar fallback for non-16-byte-aligned cases
- Decode-specific GEMM layout for small `M`
- cuBLAS backend comparison

A key issue was alignment. GPT-2 hidden dimensions are mostly 4-aligned, but the vocabulary size `50257` is not divisible by 4. Therefore, vectorized loads require an alignment-aware fallback path.

---

### 4. FlashAttention-style Prefill

During prefill, attention must process the full prompt. A naive implementation materializes the `[seq_len, seq_len]` attention score matrix in global memory.

This project implements a FlashAttention-style prefill kernel with:

- Q/K/V tiling
- Online softmax
- Causal masking
- Tile-wise accumulation
- No full attention matrix materialization

This is used only for the prefill stage. Decode uses a different paged attention kernel because decode reads historical K/V from the KV cache.

---

### 5. Paged KV Cache

A contiguous KV cache layout is simple, but it becomes inefficient when requests have different lengths. It can lead to fragmentation and makes admission control difficult.

`mini-llm` uses a paged KV cache layout.

```text
Logical block table per request
[0] [1] [2]
 |   |   |
 v   v   v
Physical KV blocks
[7] [3] [12]
```

For a token position `t`:

```text
logical_block = t / BLOCK_SIZE
offset        = t % BLOCK_SIZE
physical_blk  = block_table[logical_block]
K/V address   = pool[physical_blk, offset]
```

This allows each request to see a logically contiguous KV cache while physical memory is allocated in fixed-size blocks.

---

### 6. KV Block Manager and Swapping

The runtime separates KV cache storage from allocation policy.

| Component         | Responsibility                                                 |
| ----------------- | -------------------------------------------------------------- |
| `Pool`            | Owns GPU KV cache memory                                       |
| `CpuPool`         | Owns CPU-side swap memory                                      |
| `PhysicalBlock`   | Represents a GPU KV block                                      |
| `BlockManager`    | Allocates/frees GPU and CPU blocks                             |
| `PagedKVCache`    | Stores per-request logical block tables                        |
| `RealKvAllocator` | Performs prefill/decode admission, free, swap-out, and swap-in |

When GPU KV blocks are insufficient, KV cache can be moved to CPU memory. Swap-in uses a staging buffer:

```text
CPU contiguous KV blocks
  -> GPU staging buffer
  -> scatter into newly allocated GPU KV blocks
```

This reduces per-block transfer overhead compared with launching one copy per block.

---

### 7. Scheduler

The scheduler manages request lifecycle and batch execution.

Main scheduling ideas:

- FCFS-based request admission
- Separate prefill and decode queues
- KV block admission before prefill
- Additional block admission at decode block boundaries
- Deferred queues when KV blocks are temporarily unavailable
- Swap-out queue for CPU-resident KV cache
- Batch execution up to `MAX_BATCH_NUM`
- Response notification to the libuv event loop via `uv_async_t`

Current queue set:

```text
waiting_prefill_queue_
prefill_queue_
decode_queue_
finish_queue_
deferred_prefill_queue_
deferred_decode_queue_
swap_out_queue_
cancel_queue_
```

---

### 8. libuv Server Runtime

The server runtime is built with libuv.

It currently accepts newline-delimited token ID sequences over TCP. Each line is parsed as one request.

Example request:

```text
15496 11 616 1438 318
```

The server streams generated token IDs back to the client, one token per line.

Topics explored in the server runtime:

- Event-loop based network I/O
- Non-blocking TCP accept/read/write
- Request queue between event loop and scheduler thread
- `uv_async_t` notification from scheduler to event loop
- Graceful scheduler shutdown using queue close and thread join

---

## Repository Structure

```text
include/
  constants.h
  kernels/
    gemm.cuh
    linear.cuh
    layernorm.cuh
    gelu.cuh
    residual.cuh
    argmax.cuh
    prefill/
      append_kv.cuh
      flashattention.cuh
    decode/
      paged_attention.cuh
  model/
    gpt2_model.cuh
  runtime/
    request.h
    response.h
    scheduler.h
    server.h
    pagekvcache.h
    block_manager.h
    pool.cuh
    cpu_pool.cuh
    real_kv_allocator.h
    inference_backend.h

src/
  kernels/
  model/
  runtime/

benchmarks/
  gpt2/
  runtime/

tests/
  cpp/
  kernels/
  model/

scripts/
  export_weights.py
  dump_gpt2_golden.py
  dump_gpt2_batch_prefill_golden.py
  dump_gpt2_batch_decode_golden.py
```

---

## Requirements

Core build:

- CMake 3.18+
- C++17 compiler
- CUDA Toolkit
- NVIDIA GPU

Optional server runtime:

- libuv development headers and library

Python scripts:

- Python 3
- `torch`
- `transformers`
- `numpy`

---

## Build

### Build CUDA kernels, model, tests, and benchmarks

```bash
cmake -S . -B build -DBUILD_RUNTIME=OFF -DBUILD_CUDA_TESTS=ON
cmake --build build -j
```

### Build the libuv server with the real GPT-2 backend

```bash
cmake -S . -B build -DBUILD_RUNTIME=ON -DBUILD_CUDA_TESTS=ON
cmake --build build -j
```

If libuv is installed under `$HOME/.local`:

```bash
cmake -S . -B build \
  -DBUILD_RUNTIME=ON \
  -DBUILD_CUDA_TESTS=ON \
  -DCMAKE_PREFIX_PATH=$HOME/.local \
  -DCMAKE_INCLUDE_PATH=$HOME/.local/include \
  -DCMAKE_LIBRARY_PATH=$HOME/.local/lib

cmake --build build -j
```

### Build the server with mock backend only

This is useful for testing the runtime without CUDA.

```bash
cmake -S . -B build \
  -DBUILD_RUNTIME=ON \
  -DUSE_MOCK_BACKEND=ON \
  -DBUILD_CUDA_TESTS=OFF

cmake --build build -j
```

---

## Prepare GPT-2 Weights

Export GPT-2 small weights from Hugging Face:

```bash
pip install torch transformers numpy
python scripts/export_weights.py
```

This creates binary weight files under `weights/`, which are loaded by the C++/CUDA model.

---

## Run Tests

Run all registered tests:

```bash
ctest --test-dir build --output-on-failure
```

Run selected tests manually:

```bash
./build/test_gemm
./build/test_layernorm
./build/test_gelu
./build/test_embedding
./build/test_append_kv
./build/test_flashattention
./build/test_gpt2_prefill_logits
./build/test_gpt2_decode_logits
./build/test_gpt2_batch_prefill_logits
./build/test_gpt2_batch_decode_logits
```

On a Slurm-based machine:

```bash
srun --gres=gpu:1 ctest --test-dir build --output-on-failure
```

Some GPT-2 end-to-end tests require golden files generated by the scripts under `scripts/`.

---

## Run the Server

Build with `BUILD_RUNTIME=ON`, then run:

```bash
./build/mini_llm
```

Send a request using `nc`:

```bash
printf "15496 11 616 1438 318\n" | nc localhost 8080
```

The server returns generated token IDs line by line.

---

## Benchmarks

### Batched vs sequential GPT-2 inference

```bash
./build/bench_batch_vs_single_wall 5 8 5 1
```

Arguments:

```text
./build/bench_batch_vs_single_wall [prompt_len] [max_new_tokens] [iterations] [warmup]
```

Output columns:

```text
batch_size,prompt_len,max_new_tokens,iterations,mode,avg_total_ms,tokens_per_sec
```

### Prefill/decode by prompt length

```bash
./build/bench_prefill_decode_by_len
```

### KV swap benchmarks

```bash
./build/bench_direct_swap_in_by_blocks
./build/bench_buffered_swap_in_by_blocks
./build/bench_swap_in_prefill_decode_overlap
```

---

## Correctness Strategy

Correctness is checked at multiple levels.

| Component                    | Reference                                  |
| ---------------------------- | ------------------------------------------ |
| GEMM                         | CPU/NumPy-style expected output            |
| LayerNorm                    | CPU reference                              |
| GELU                         | CPU reference                              |
| Embedding                    | CPU reference                              |
| Append KV                    | Expected block/offset layout               |
| FlashAttention-style prefill | Naive causal attention reference           |
| GPT-2 prefill logits         | Hugging Face golden logits                 |
| GPT-2 decode logits          | Hugging Face golden logits                 |
| Batched GPT-2 logits         | Hugging Face golden logits                 |
| KV swap                      | Expected block residency and data movement |

---

## Lessons Learned

- GPU optimization should start from profiling and bottleneck analysis, not from applying techniques blindly.
- Prefill and decode need different optimization strategies.
- Small-batch decode is often limited by poor parallelism and CPU/GPU overhead.
- Removing unnecessary D2H transfers can provide large latency improvements.
- Paged KV cache reduces contiguous allocation pressure but introduces block-table lookup overhead.
- Scheduler design must explicitly handle KV memory admission failures.
- Vectorized memory access is powerful but must be alignment-safe.
- Runtime shutdown should be designed explicitly; worker threads should not be forcefully terminated.

---

## Current Limitations

- This is an educational/experimental engine, not a production inference server.
- Tokenization is not implemented in the server; requests currently use token IDs directly.
- Sampling is greedy argmax only.
- The scheduler is FCFS-based and does not yet implement advanced policies such as priority scheduling.
- Swap-in/out is still synchronous in the scheduler path.
- Tensor Core / FP16 GEMM is not implemented yet.
- vLLM is still significantly faster, especially for optimized attention and long-context prefill.

---

## Roadmap

- [ ] Implement Tensor Core FP16 GEMM
- [ ] Improve long-sequence prefill attention performance
- [ ] Add asynchronous swap-in/out with CUDA streams
- [ ] Add prefix caching
- [ ] Add tokenizer support
- [ ] Add sampling options such as temperature and top-k/top-p
- [ ] Add server concurrency benchmark
- [ ] Add more detailed profiling reports
- [ ] Explore coroutine-based runtime design

---

## References

| Topic                    | Reference                                                                                                              | How it relates to this project                                                                  |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Transformer architecture | Ashish Vaswani et al., **Attention Is All You Need**, NeurIPS 2017                                                     | Transformer block structure, scaled dot-product attention, residual connections, LayerNorm, FFN |
| LLM serving scheduler    | Gyeong-In Yu et al., **Orca: A Distributed Serving System for Transformer-Based Generative Models**, OSDI 2022         | Iteration-level scheduling, prefill/decode distinction, continuous batching ideas               |
| Paged KV cache           | vLLM / PagedAttention                                                                                                  | Block-based KV cache management and non-contiguous decode attention                             |
| IO-aware attention       | Tri Dao et al., **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**, 2022                  | Tiled prefill attention, online softmax, reduced HBM traffic                                    |
| Online softmax           | Maxim Milakov and Natalia Gimelshein, **Online Normalizer Calculation for Softmax**, 2018                              | Numerically stable online softmax and memory-pass reduction                                     |
| Performance modeling     | Samuel Williams, Andrew Waterman, and David Patterson, **Roofline: An Insightful Visual Performance Model**, CACM 2009 | Operational intensity, memory-bound vs compute-bound analysis                                   |
| GPU architecture         | Zhe Jia et al., **Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking**, 2018                           | CUDA memory hierarchy and architecture-aware optimization                                       |

---

## Summary

`mini-llm` is a bottom-up GPT-2 inference engine built to study LLM serving internals.

The main focus is:

- CUDA kernel implementation
- GPU memory management
- Paged KV cache design
- Prefill/decode scheduling
- CPU/GPU data movement optimization
- Asynchronous server runtime design
- Correctness and benchmark-driven development
