import os
os.environ.setdefault("VLLM_USE_V1", "0")

import sys
import time
import torch

from vllm import LLM, SamplingParams


BASE_TOKENS = [15496, 11, 616, 1438, 318]
# "Hello, my name is" 비슷한 token id 시퀀스


def make_prompt(prompt_len):
    return [BASE_TOKENS[i % len(BASE_TOKENS)] for i in range(prompt_len)]


def run_once(llm, batch_size, prompt_len, max_new_tokens):
    prompts = [
        {"prompt_token_ids": make_prompt(prompt_len)}
        for _ in range(batch_size)
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        ignore_eos=True,
        detokenize=False,
    )

    torch.cuda.synchronize()
    start = time.perf_counter()

    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=False,
    )

    torch.cuda.synchronize()
    end = time.perf_counter()

    total_output_tokens = sum(
        len(out.outputs[0].token_ids)
        for out in outputs
    )

    return end - start, total_output_tokens


def benchmark(llm, batch_size, prompt_len, max_new_tokens, iterations, warmup):
    for _ in range(warmup):
        run_once(llm, batch_size, prompt_len, max_new_tokens)

    total_time = 0.0
    total_tokens = 0

    for _ in range(iterations):
        elapsed, tokens = run_once(
            llm,
            batch_size,
            prompt_len,
            max_new_tokens,
        )

        total_time += elapsed
        total_tokens += tokens

    avg_time_ms = total_time / iterations * 1000.0
    tokens_per_sec = total_tokens / total_time

    print(
        f"{batch_size},"
        f"{prompt_len},"
        f"{max_new_tokens},"
        f"{iterations},"
        f"{avg_time_ms:.4f},"
        f"{tokens_per_sec:.4f}"
    )


def main():
    prompt_len = int(sys.argv[1]) if len(sys.argv) >= 2 else 5
    max_new_tokens = int(sys.argv[2]) if len(sys.argv) >= 3 else 8
    iterations = int(sys.argv[3]) if len(sys.argv) >= 4 else 5
    warmup = int(sys.argv[4]) if len(sys.argv) >= 5 else 1

    llm = LLM(
        model="openai-community/gpt2",
        dtype="float",
        max_model_len=1024,
        gpu_memory_utilization=0.6,
        disable_log_stats=True,
        enforce_eager=False,
        skip_tokenizer_init=True,
    )

    print("batch_size,prompt_len,max_new_tokens,iterations,avg_total_ms,tokens_per_sec")

    for batch_size in [1, 2, 4, 8, 16]:
        benchmark(
            llm,
            batch_size,
            prompt_len,
            max_new_tokens,
            iterations,
            warmup,
        )


if __name__ == "__main__":
    main()
