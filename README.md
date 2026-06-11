# mini-llm

CUDA Kernel, Paged KV Cache, vLLM-style Scheduler를 직접 구현한 GPT-2 기반 mini LLM Serving Engine입니다.

이 프로젝트는 GPT-2 inference를 단순 forward pass가 아니라 **LLM serving system** 관점에서 구현하기 위해 시작했습니다. CUDA kernel, KV cache memory manager, prefill/decode scheduler, 그리고 server runtime을 직접 설계하며 LLM serving에서 발생하는 memory, scheduling, batching 병목을 분석했습니다.

---

## Overview

LLM serving은 일반적인 batch inference와 다릅니다.

- Request마다 prompt 길이와 generation 길이가 다릅니다.
- Prefill 단계와 decode 단계의 연산 특성이 다릅니다.
- Decode 과정에서 KV cache가 token 단위로 계속 증가합니다.
- KV cache를 request마다 contiguous하게 할당하면 fragmentation과 admission 문제가 발생합니다.
- 여러 request를 효율적으로 처리하려면 scheduler가 GPU memory 상태와 batch 구성 모두를 고려해야 합니다.

`mini-llm`은 이러한 문제를 이해하기 위해 GPT-2 style model을 대상으로 다음 구성요소를 직접 구현한 프로젝트입니다.

- GPT-2 inference pipeline
- Custom CUDA kernels
- FlashAttention-style prefill attention
- Paged decode attention
- Paged KV cache
- GPU memory pool and block manager
- Prefill/decode separated scheduler
- Deferred queue for KV block shortage
- Asynchronous server runtime
- NumPy / Hugging Face based correctness tests

---

## Architecture

```text
Client
  ↓
Server / Event Loop
  ↓
Request Queue
  ↓
Scheduler
  ├── Waiting Prefill Queue
  ├── Prefill Queue
  ├── Decode Queue
  ├── Deferred Prefill Queue
  ├── Deferred Decode Queue
  ├── Finish Queue
  └── Cancel Queue
  ↓
GPT2Model
  ↓
CUDA Kernels
  ├── GEMM
  ├── LayerNorm
  ├── GELU
  ├── FlashAttention-style Prefill
  └── Paged Decode Attention
  ↓
Paged KV Cache Pool
```

### Runtime Flow

```text
New Request
  ↓
waiting_prefill_queue
  ↓ block admission
prefill_queue
  ↓ model.prefill()
decode_queue
  ↓ model.decode()
finish_queue
```

KV block이 부족한 경우 request를 즉시 실패시키지 않고 deferred queue로 이동시킵니다.

```text
Prefill admission failed
  → deferred_prefill_queue

Decode needs new KV block but memory is insufficient
  → deferred_decode_queue
```

---

## Key Features

### 1. Custom CUDA Kernels

| Kernel                       | Key Ideas                                                                     | Purpose                                            |
| ---------------------------- | ----------------------------------------------------------------------------- | -------------------------------------------------- |
| GEMM                         | Shared memory tiling, register blocking, vectorized load with scalar fallback | Linear projection and logits computation           |
| LayerNorm                    | Block-level reduction                                                         | Row-wise normalization                             |
| GELU                         | Elementwise CUDA kernel                                                       | FFN activation                                     |
| FlashAttention-style Prefill | Tiling, online softmax, causal masking                                        | Efficient prompt attention                         |
| Append KV                    | Block/offset based KV write                                                   | Store K/V into paged KV cache                      |
| Paged Decode Attention       | Block table based K/V lookup                                                  | Decode-time attention over non-contiguous KV cache |

### 2. GEMM Optimization

GEMM kernel은 다음 최적화를 적용했습니다.

- Shared memory tiling으로 global memory traffic 감소
- `float4` vectorized load로 aligned memory access 최적화
- Row stride가 4의 배수가 아닌 경우를 위한 scalar fallback

`float4` load는 16-byte alignment가 보장될 때만 안전합니다. GPT-2의 hidden dimension은 대부분 4의 배수지만, vocab size `50257`은 4의 배수가 아니기 때문에 final vocab projection에서는 alignment 문제가 발생할 수 있습니다. 따라서 aligned address에서는 vectorized load를 사용하고, 그렇지 않은 경우 scalar load로 fallback하도록 구현했습니다.

### 3. FlashAttention-style Prefill

Prefill 단계에서는 prompt 전체에 대해 causal attention을 수행합니다. Naive attention은 `[seq_len, seq_len]` score matrix를 global memory에 materialize하기 때문에 memory traffic이 커집니다.

이를 줄이기 위해 다음 아이디어를 적용했습니다.

- Q/K/V tile을 shared memory에 적재
- Online softmax로 score matrix materialization 제거
- Causal mask 적용
- Tile 단위 누적을 통해 memory traffic 감소

### 4. Paged Decode Attention

Decode 단계에서는 request마다 query는 한 token이지만, K/V는 과거 모든 token을 참조해야 합니다.

KV cache를 contiguous buffer로 관리하면 request 길이가 다양할 때 fragmentation이 발생합니다. 이를 해결하기 위해 vLLM의 PagedAttention 아이디어를 참고하여 KV cache를 block 단위로 관리했습니다.

```text
Logical block table per request
[0] [1] [2]
 ↓   ↓   ↓
Physical KV blocks
[7] [3] [12]
```

Decode attention kernel은 K/V를 별도의 contiguous buffer로 gather하지 않고, block table을 통해 physical KV block을 직접 참조합니다.

```text
token index t
  ↓
logical_block = t / BLOCK_SIZE
offset        = t % BLOCK_SIZE
  ↓
physical_block = block_table[logical_block]
  ↓
K/V address = pool[physical_block, offset]
```

---

## Paged KV Cache

### Components

| Component       | Responsibility                                                         |
| --------------- | ---------------------------------------------------------------------- |
| `Pool`          | Owns GPU KV cache memory                                               |
| `PhysicalBlock` | Represents a physical KV block                                         |
| `BlockManager`  | Allocates and frees physical blocks                                    |
| `PagedKVCache`  | Stores request-local logical block table                               |
| `Request`       | Tracks tokens, prompt length, generation state, and per-layer KV cache |

### Why separate Pool and BlockManager?

`Pool`은 GPU memory storage를 소유하고, `BlockManager`는 allocation policy를 담당합니다. 두 책임을 분리하면 향후 swapping, prefix caching, priority scheduling 등을 추가하기 쉽습니다.

---

## Scheduler

Scheduler는 request lifecycle과 batch execution을 관리합니다.

### Queues

```text
waiting_prefill_queue_
prefill_queue_
decode_queue_
finish_queue_
deferred_prefill_queue_
deferred_decode_queue_
cancel_queue_
```

### Scheduling Policy

현재 scheduler는 FCFS 기반으로 동작합니다.

- New request는 `waiting_prefill_queue_`에 들어갑니다.
- Prefill 전에 필요한 KV block을 admission합니다.
- Prefill이 끝나면 첫 output token을 생성하고 decode queue로 이동합니다.
- Decode 단계에서는 새 KV block이 필요한 boundary에서 block admission을 수행합니다.
- KV block이 부족하면 deferred queue로 이동합니다.
- Request가 finish되면 사용한 KV block을 반환합니다.

### Prefill vs Decode

| Stage        | Characteristics            | Main Bottleneck                        | Optimization                                       |
| ------------ | -------------------------- | -------------------------------------- | -------------------------------------------------- |
| Prefill      | Full prompt processing     | Attention and GEMM compute             | FlashAttention-style tiling, batched execution     |
| Decode       | One token per step         | KV cache read and small-batch overhead | Paged KV cache, block table lookup, batched decode |
| Final logits | Hidden to vocab projection | Large vocab GEMM                       | GEMM optimization, alignment-aware load            |

---

## Server Runtime

The server runtime is designed around asynchronous I/O.

Topics explored in this project:

- C10K problem
- Blocking I/O vs non-blocking I/O
- `select` / `poll` / `epoll`
- libuv event loop
- Scheduler thread and event loop thread coordination
- Graceful shutdown with queue close and condition variable notification

The server can be built optionally depending on whether libuv development headers are available.

---

## Correctness Tests

Kernel correctness is verified against CPU or Python references.

| Component              | Reference              | Status      |
| ---------------------- | ---------------------- | ----------- |
| GEMM                   | NumPy matmul           | Pass        |
| LayerNorm              | NumPy implementation   | Pass        |
| GELU                   | NumPy formula          | Pass        |
| Embedding              | CPU reference          | Pass        |
| Append KV              | CPU expected KV layout | Pass        |
| FlashAttention Prefill | Naive causal attention | In progress |
| Paged Decode Attention | CPU paged attention    | In progress |
| GPT-2 logits           | Hugging Face GPT-2     | In progress |

---

## Benchmarks

Benchmark results will be updated as implementation stabilizes.

### Kernel Benchmark Plan

| Benchmark                          | Variables                  | Metrics            |
| ---------------------------------- | -------------------------- | ------------------ |
| GEMM scalar vs vectorized fallback | M, N, K, alignment         | latency, max error |
| LayerNorm                          | batch size, hidden dim     | latency            |
| FlashAttention Prefill             | sequence length            | latency            |
| Paged Decode Attention             | batch size, context length | latency            |
| Append KV                          | token count, block size    | latency            |

### Serving Benchmark Plan

| Benchmark           | Variables              | Metrics                              |
| ------------------- | ---------------------- | ------------------------------------ |
| Batch size scaling  | batch = 1, 2, 4, 8, 16 | tokens/sec, latency                  |
| Prefill vs Decode   | prompt len, decode len | prefill latency, decode step latency |
| Scheduler admission | KV block pressure      | deferred count, success rate         |
| Server concurrency  | number of clients      | throughput, p95 latency              |

---

## Build

### Build without server runtime

If libuv development headers are not available, build only kernels/model/tests.

```bash
cmake -S . -B build -DBUILD_RUNTIME=OFF
cmake --build build -j
```

### Build with server runtime

If libuv is installed:

```bash
cmake -S . -B build -DBUILD_RUNTIME=ON
cmake --build build -j
```

If libuv is installed under `$HOME/.local`:

```bash
cmake -S . -B build \
  -DBUILD_RUNTIME=ON \
  -DCMAKE_PREFIX_PATH=$HOME/.local \
  -DCMAKE_INCLUDE_PATH=$HOME/.local/include \
  -DCMAKE_LIBRARY_PATH=$HOME/.local/lib

cmake --build build -j
```

---

## Test

Run all registered tests:

```bash
ctest --test-dir build --output-on-failure
```

Run individual tests:

```bash
./build/test_gemm
./build/test_layernorm
./build/test_gelu
./build/test_embedding
./build/test_append_kv
```

If the machine uses a scheduler such as Slurm:

```bash
srun --gres=gpu:1 ctest --test-dir build --output-on-failure
```

---

## Project Structure

```text

include/
├── constants.h
├── kernels/
│   │   ├── gemm.cuh
│   │   ├── layernorm.cuh
│   │   ├── gelu.cuh
│   │   ├── embedding.cuh
│   │   ├── residual.cuh
│   │   ├── warp_reduction.cuh
│   │   ├── prefill/
│   │   │   ├── append_kv.cuh
│   │   │   └── flashattention.cuh
│   │   └── decode/
│   │       └── paged_attention.cuh
│   ├── model/
│   │   └── gpt2_model.cuh
│   └── runtime/
│       ├── request.h
│       ├── response.h
│       ├── pagekvcache.h
│       ├── block.h
│       ├── block_manager.h
│       ├── pool.cuh
│       ├── mutex_queue.h
│       ├── scheduler.h
│       └── server.h
├── src/
├── kernels/
├── model/
└── runtime/
tests/
    └── cpp/
```

---

## Lessons Learned

- GPU kernel optimization should be guided by memory access pattern and operational intensity, not only by adding more arithmetic.
- `float4` vectorized load is only safe when 16-byte alignment is guaranteed.
- GPT-2 hidden dimensions are mostly 4-aligned, but vocab size is not necessarily aligned.
- Prefill and decode have different bottlenecks, so they should be scheduled and optimized differently.
- Paged KV cache reduces contiguous allocation pressure but introduces block table lookup overhead.
- Scheduler should handle memory admission explicitly instead of assuming KV allocation always succeeds.
- Worker threads should not be forcefully terminated; graceful shutdown should be implemented through queue close and condition variable notification.

---

## Roadmap

- [ ] Complete paged decode attention correctness test
- [ ] Compare FlashAttention-style prefill with naive causal attention
- [ ] Compare GPT-2 logits with Hugging Face reference
- [ ] Add batch size benchmark
- [ ] Add server concurrency benchmark
- [ ] Add Tensor Core GEMM implementation
- [ ] Add KV cache swapping
- [ ] Add prefix caching
- [ ] Explore coroutine-based server runtime

---

## References

This project was implemented while reading and reproducing ideas from the following papers and technical reports.

| Topic                    | Reference                                                                                                                                                                        | How it relates to this project                                                                                                                                       |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Transformer architecture | Ashish Vaswani et al., **"Attention Is All You Need"**, NeurIPS 2017                                                                                                             | Baseline Transformer/GPT-style architecture, scaled dot-product attention, multi-head attention, positional embedding, residual connection, LayerNorm, FFN structure |
| LLM serving scheduler    | Gyeong-In Yu et al., **"Orca: A Distributed Serving System for Transformer-Based Generative Models"**, OSDI 2022                                                                 | Motivation for iteration-level scheduling, prefill/decode distinction, continuous batching, and request-level vs iteration-level scheduling                          |
| IO-aware attention       | Tri Dao et al., **"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"**, 2022                                                                          | Prefill attention optimization, tiling, avoiding attention matrix materialization, reducing HBM traffic, online softmax based accumulation                           |
| Online softmax           | Maxim Milakov and Natalia Gimelshein, **"Online Normalizer Calculation for Softmax"**, 2018                                                                                      | Numerically stable online softmax, reducing memory passes, softmax/top-k fusion motivation                                                                           |
| Performance model        | Samuel Williams, Andrew Waterman, and David Patterson, **"Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures"**, CACM 2009 | Kernel performance analysis using operational intensity, peak FLOPS, memory bandwidth, and ridge point                                                               |
| GPU architecture         | Zhe Jia et al., **"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking"**, 2018                                                                                   | GPU memory hierarchy, shared memory, L1/L2 behavior, register bank conflict, and architecture-aware CUDA optimization                                                |

## Summary

`mini-llm` is a minimal LLM serving engine built to understand the internals of modern LLM inference systems.

This project focuses on:

- CUDA kernel implementation
- GPU memory management
- KV cache paging
- Prefill/decode scheduling
- Asynchronous server runtime
- Correctness and benchmark-driven development
