"""my-vLLM 可复现在线服务压测器（仅使用Python标准库）。"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import platform
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values, default=0.0),
        "max": max(values, default=0.0),
    }


class NvidiaSmiSampler:
    def __init__(self, gpu_index: int, interval_ms: int = 100):
        self.gpu_index = gpu_index
        self.interval_ms = interval_ms
        self.samples: list[dict[str, float]] = []
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        command = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self.interval_ms}",
        ]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        def consume() -> None:
            assert self.process is not None and self.process.stdout is not None
            for line in self.process.stdout:
                try:
                    memory, utilization, power = (
                        float(item.strip()) for item in line.split(",")
                    )
                except (ValueError, TypeError):
                    continue
                self.samples.append(
                    {
                        "time": time.time(),
                        "memory_mib": memory,
                        "gpu_util_percent": utilization,
                        "power_w": power,
                    }
                )

        self.thread = threading.Thread(target=consume, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.thread is not None:
            self.thread.join(timeout=2)

    def summary(self) -> dict[str, Any]:
        memories = [sample["memory_mib"] for sample in self.samples]
        utilizations = [sample["gpu_util_percent"] for sample in self.samples]
        powers = [sample["power_w"] for sample in self.samples]
        return {
            "sample_interval_ms": self.interval_ms,
            "num_samples": len(self.samples),
            "memory_mib": distribution(memories),
            "gpu_util_percent": distribution(utilizations),
            "power_w": distribution(powers),
        }


def load_workload(path: Path) -> tuple[list[dict[str, Any]], str]:
    content = path.read_bytes()
    records = [json.loads(line) for line in content.decode().splitlines() if line]
    return records, hashlib.sha256(content).hexdigest()


def send_request(endpoint: str, record: dict[str, Any]) -> dict[str, Any]:
    body = {
        "model": "Qwen2.5-7B-Instruct",
        "messages": record["messages"],
        "max_tokens": record["max_tokens"],
        "ignore_eos": record.get("ignore_eos", True),
        "return_metrics": True,
    }
    encoded = json.dumps(body, ensure_ascii=False).encode()
    request = urllib.request.Request(
        endpoint,
        data=encoded,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=3600) as response:
        payload = json.loads(response.read())
    finish = time.perf_counter()
    metrics = payload["benchmark_metrics"]
    if metrics["prompt_tokens"] != record["prompt_tokens"]:
        raise RuntimeError(
            f"{record['request_id']} prompt_tokens="
            f"{metrics['prompt_tokens']} != {record['prompt_tokens']}"
        )
    return {
        "request_id": record["request_id"],
        "client_e2e_ms": (finish - start) * 1000,
        **metrics,
    }


def run_requests(
    endpoint: str,
    records: list[dict[str, Any]],
    concurrency: int,
) -> tuple[list[dict[str, Any]], float]:
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send_request, endpoint, record) for record in records]
        results = [future.result() for future in futures]
    return results, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:13311/v1/chat/completions")
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records, workload_sha256 = load_workload(args.workload)
    if args.warmup_requests:
        run_requests(args.endpoint, records[: args.warmup_requests], 1)

    sampler = NvidiaSmiSampler(args.gpu)
    sampler.start()
    try:
        results, duration_s = run_requests(
            args.endpoint, records, args.concurrency
        )
    finally:
        sampler.stop()

    total_prompt_tokens = sum(item["prompt_tokens"] for item in results)
    total_output_tokens = sum(item["output_tokens"] for item in results)
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "workload": str(args.workload),
        "workload_sha256": workload_sha256,
        "num_requests": len(results),
        "concurrency": args.concurrency,
        "duration_s": duration_s,
        "request_throughput_rps": len(results) / duration_s,
        "prompt_throughput_tokens_per_s": total_prompt_tokens / duration_s,
        "output_throughput_tokens_per_s": total_output_tokens / duration_s,
        "total_token_throughput_tokens_per_s": (
            total_prompt_tokens + total_output_tokens
        ) / duration_s,
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "ttft_ms": distribution([item["ttft_ms"] for item in results]),
        "tpot_ms": distribution([item["tpot_ms"] for item in results]),
        "client_e2e_ms": distribution(
            [item["client_e2e_ms"] for item in results]
        ),
        "engine_e2e_ms": distribution(
            [item["e2e_engine_ms"] for item in results]
        ),
        "generation_ms": distribution(
            [item["generation_ms"] for item in results]
        ),
        "preemptions_total": sum(item["num_preemptions"] for item in results),
        "gpu": sampler.summary(),
    }
    output = {"summary": summary, "requests": results, "gpu_samples": sampler.samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
