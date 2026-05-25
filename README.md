# mini-llm

A minimal LLM inference server written in C++/CUDA, featuring
WMMA-based GEMM, paged KV cache, continuous batching scheduler,
and HTTP serving for GPT-2 style models.

## Features
- GPT-2 inference in C++/CUDA
- Custom WMMA GEMM kernel
- Paged KV cache allocator
- Prefill/decode scheduler
- HTTP inference server
- Benchmark and profiling reports

## Architecture
Client → HTTP Server → Scheduler → Prefill/Decode Engine → CUDA Kernels → KV Cache

## Performance


## Roadmap
- Paged attention kernel
- FlashAttention-lite
- INT8 quantization
- Continuous batching
