#!/usr/bin/env python3
"""Measure HTTP serving latency for a running my_vllm TP/PP deployment.

The current API is non-streaming, so this tool reports end-to-end latency and
output throughput. TTFT needs the streaming API and is intentionally emitted as
``null`` rather than inferred from a completed response.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percent / 100))
    return ordered[index]


def request_once(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    tokenizer=None,
) -> tuple[float, int, int]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
    ).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.load(response)
    elapsed = time.perf_counter() - started
    output = body["choices"][0]["message"]["content"]
    output_tokens = len(tokenizer.encode(output).ids) if tokenizer else max_tokens
    return elapsed, len(output), output_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="请简要介绍张量并行与流水线并行的区别。")
    parser.add_argument(
        "--repeat-word",
        default=None,
        help="设置后用 (' ' + word) * repeat_count 构造固定长 prompt content",
    )
    parser.add_argument("--repeat-count", type=int, default=0)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="tokenizer.json 路径；用于报告真实 completion token 数",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()

    if args.requests < 1 or args.concurrency < 1:
        raise ValueError("--requests 和 --concurrency 必须大于 0")

    tokenizer = None
    if args.tokenizer:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(args.tokenizer)
    prompt = args.prompt
    if args.repeat_word is not None:
        prompt = (" " + args.repeat_word) * args.repeat_count

    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    for _ in range(args.warmup):
        request_once(endpoint, args.model, prompt, args.max_tokens, tokenizer)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                request_once,
                endpoint,
                args.model,
                prompt,
                args.max_tokens,
                tokenizer,
            )
            for _ in range(args.requests)
        ]
        results = [future.result() for future in futures]
    wall_time = time.perf_counter() - started

    latencies = [latency for latency, _, _ in results]
    output_units = sum(length for _, length, _ in results)
    output_tokens = sum(tokens for _, _, tokens in results)
    prompt_tokens = (
        len(tokenizer.encode("[user]: " + prompt).ids) if tokenizer else None
    )
    summary = {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "prompt_tokens": prompt_tokens,
        "wall_time_s": round(wall_time, 4),
        "e2e_latency_ms": {
            "mean": round(statistics.mean(latencies) * 1000, 2),
            "p50": round(percentile(latencies, 50) * 1000, 2),
            "p95": round(percentile(latencies, 95) * 1000, 2),
        },
        "output_units_per_s": round(output_units / wall_time, 2),
        "output_tokens": output_tokens,
        "output_token_throughput": round(output_tokens / wall_time, 2),
        "ttft_ms": None,
        "note": "TTFT requires streaming API; output token throughput uses tokenizer.json when supplied.",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
