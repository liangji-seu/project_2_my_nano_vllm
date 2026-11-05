#!/usr/bin/env python3
"""Small, repeatable HTTP benchmark matrix for the local chat-completions server."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import socket
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encode_ids(tokenizer: Any, text: str, special: bool) -> list[int]:
    result = tokenizer.encode(text, add_special_tokens=special)
    return list(getattr(result, "ids", result))


def render_prompt(content: str) -> str:
    return "[user]: " + content


def _synthetic_one(tokenizer: Any, target: int, index: int, seed: int = 0) -> tuple[str, str, int]:
    rng = random.Random(f"{seed}:{index}")
    prefix = f"request-{index:08d}-{rng.getrandbits(64):016x}: unique early prefix."
    # Qwen tokenizers normally encode ` hello` as one token.  We still measure
    # every candidate and never rely on that assumption for correctness.
    base = len(encode_ids(tokenizer, render_prompt(prefix), True))
    if base > target:
        raise ValueError(f"cannot create synthetic request {index}: target {target} is shorter than its unique prefix ({base} tokens)")
    probe = len(encode_ids(tokenizer, render_prompt(prefix + " hello"), True)) - base
    estimate = max(0, target - base) if probe <= 0 else max(0, (target - base + probe - 1) // probe)
    candidates = [estimate]
    for distance in range(1, 65):
        candidates.extend((estimate - distance, estimate + distance))
    for n in candidates:
        if n < 0:
            continue
        content = prefix + (" hello" * n)
        rendered = render_prompt(content)
        size = len(encode_ids(tokenizer, rendered, True))
        if size == target:
            return content, f"synthetic-{index}", size
    raise ValueError(
        f"cannot create synthetic request {index} with exactly {target} rendered tokens; "
        "try a different input length or tokenizer"
    )


def synthetic_records(tokenizer: Any, target: int, count: int, seed: int = 0) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("request count must be positive")
    out = []
    for i in range(count):
        content, source_id, actual = _synthetic_one(tokenizer, target, i, seed)
        out.append({"content": content, "source_id": source_id, "record_index": i, "input_tokens": actual})
    return out


def load_records(path: Path, kind: str) -> list[dict[str, Any]]:
    if kind == "jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("dataset must contain a JSON array (or JSONL objects)")
    result = []
    for index, row in enumerate(records):
        content = None
        if kind == "jsonl":
            if isinstance(row, dict) and isinstance(row.get("prompt"), str):
                content = row["prompt"]
        else:
            if isinstance(row, dict):
                conv = row.get("conversations")
                if isinstance(conv, list):
                    for turn in conv:
                        if isinstance(turn, dict) and str(turn.get("from", "")).lower() in {"human", "user"} and isinstance(turn.get("value"), str):
                            content = turn["value"]
                            break
        if content is not None:
            result.append({"content": content, "source_id": str(row.get("id", index)) if isinstance(row, dict) else str(index), "record_index": index})
    return result


def select_real_records(records: list[dict[str, Any]], tokenizer: Any, target: int, count: int, tolerance: float, seed: int) -> list[dict[str, Any]]:
    eligible = []
    lo, hi = target * (1 - tolerance), target * (1 + tolerance)
    for row in records:
        rendered = render_prompt(row["content"])
        actual = row.get("input_tokens")
        if actual is None:
            actual = len(encode_ids(tokenizer, rendered, True))
        if lo <= actual <= hi:
            copy = dict(row)
            copy["input_tokens"] = actual
            eligible.append(copy)
    if len(eligible) < count:
        raise ValueError(f"dataset has {len(eligible)} eligible records for target {target}, need {count}; data is never repeated")
    return random.Random(seed).sample(eligible, count)


def prepare_workload(tokenizer: Any, dataset: str, records: list[dict[str, Any]] | None, target: int, count: int, tolerance: float, seed: int, output_budget: int, max_model_len: int) -> list[dict[str, Any]]:
    if dataset == "synthetic":
        chosen = synthetic_records(tokenizer, target, count, seed)
    else:
        chosen = select_real_records(records or [], tokenizer, target, count, tolerance, seed)
    for row in chosen:
        if row["input_tokens"] + output_budget > max_model_len:
            raise ValueError(f"input tokens + output budget exceeds max-model-len ({row['input_tokens']} + {output_budget} > {max_model_len})")
        row["prompt"] = render_prompt(row["content"])
        row["prompt_hash"] = sha256_bytes(row["prompt"].encode())
    return chosen


def _request(base_url: str, model: str, row: dict[str, Any], output_budget: int, timeout: float) -> dict[str, Any]:
    payload = json.dumps({"model": model, "messages": [{"role": "user", "content": row["content"]}], "max_tokens": output_budget, "stream": False}).encode()
    request = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    result: dict[str, Any] = {"source_id": row["source_id"], "prompt_hash": row["prompt_hash"], "input_tokens": row["input_tokens"], "status": "error", "http_status": None, "latency_ms": None, "estimated_output_tokens": None, "response_chars": None, "response_text_sha256": None, "error": None, "_text": None}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            result["http_status"] = response.status
        result["latency_ms"] = (time.perf_counter() - started) * 1000
        parsed = json.loads(body.decode("utf-8"))
        choices = parsed.get("choices") if isinstance(parsed, dict) else None
        message = choices[0].get("message") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str):
            raise ValueError("malformed response: choices[0].message.content must be a string")
        result.update(status="success", _text=text, response_chars=len(text), response_text_sha256=sha256_bytes(text.encode()))
    except urllib.error.HTTPError as exc:
        exc.read()
        result["http_status"] = exc.code
        result["latency_ms"] = (time.perf_counter() - started) * 1000
        result["error"] = f"HTTP {exc.code}"
    except Exception as exc:  # network errors and malformed JSON are request failures
        result["latency_ms"] = (time.perf_counter() - started) * 1000
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def run_requests(base_url: str, model: str, workload: list[dict[str, Any]], output_budget: int, concurrency: int, timeout: float, tokenizer: Any, warmup: int = 0) -> tuple[list[dict[str, Any]], float]:
    if warmup:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            warm = list(pool.map(lambda r: _request(base_url, model, r, output_budget, timeout), workload[:warmup]))
        if any(x["status"] != "success" for x in warm):
            raise RuntimeError("warmup request failed; measured case was not run")
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_request, base_url, model, row, output_budget, timeout) for row in workload]
        for future in as_completed(futures):
            results.append(future.result())
    wall = time.perf_counter() - started
    # Retokenization intentionally occurs after the wall timer.
    for result in results:
        result["request_count"] = len(workload)
        if result["status"] == "success":
            result["estimated_output_tokens"] = len(encode_ids(tokenizer, result.pop("_text"), False))
        else:
            result.pop("_text", None)
    return results, wall


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * p
    low, high = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (pos - low)


def workload_sha256(workload: list[dict[str, Any]], output_budget: int) -> str:
    value = {"prompts": [row["prompt"] for row in workload], "output_budget": output_budget}
    return sha256_bytes(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def summarize(results: list[dict[str, Any]], wall: float, target: int, count: int, concurrency: int, output_budget: int, repeat: int, workload_hash: str, input_lengths: list[int]) -> dict[str, Any]:
    successes = [r for r in results if r["status"] == "success"]
    latencies = [r["latency_ms"] for r in successes]
    outputs = [r["estimated_output_tokens"] for r in successes]
    errors = len(results) - len(successes)
    return {"repeat": repeat, "input_target": target, "output_budget": output_budget, "concurrency": concurrency, "requested": len(results), "successes": len(successes), "errors": errors, "error_rate": errors / len(results) if results else 0.0, "wall_seconds": wall, "request_throughput_success_per_s": len(successes) / wall if wall else None, "estimated_input_tokens_per_s": sum(r["input_tokens"] for r in successes) / wall if wall else None, "estimated_output_tokens_per_s": sum(outputs) / wall if wall else None, "input_tokens_min": min(input_lengths) if input_lengths else None, "input_tokens_mean": mean(input_lengths) if input_lengths else None, "input_tokens_max": max(input_lengths) if input_lengths else None, "workload_sha256": workload_hash, "latency_mean_ms": mean(latencies) if latencies else None, "latency_p50_ms": percentile(latencies, .50), "latency_p95_ms": percentile(latencies, .95), "latency_p99_ms": percentile(latencies, .99), "generated_tokens_mean": mean(outputs) if outputs else None, "generated_tokens_min": min(outputs) if outputs else None, "generated_tokens_max": max(outputs) if outputs else None, "ttft_ms": None, "tpot_ms": None, "itl_ms": None, "valid": errors == 0}


def load_tokenizer(path: str) -> Any:
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("tokenizers is required for the benchmark CLI (unit tests use a fake tokenizer)") from exc
    p = Path(path).expanduser()
    if p.is_file():
        return Tokenizer.from_file(str(p))
    if p.is_dir() and (p / "tokenizer.json").is_file():
        return Tokenizer.from_file(str(p / "tokenizer.json"))
    raise RuntimeError(f"tokenizer path must be a local tokenizer.json file or directory containing tokenizer.json: {path}")


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--dataset", choices=("synthetic", "sharegpt", "jsonl"), default="synthetic")
    p.add_argument("--dataset-path")
    p.add_argument("--requests", nargs="+", type=int, default=[32])
    p.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8])
    p.add_argument("--input-lens", nargs="+", type=int, default=[128, 512, 2048])
    p.add_argument("--length-tolerance", type=float, default=.2)
    p.add_argument("--output-lens", nargs="+", type=int, default=[128])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--label")
    p.add_argument("--output-dir", default="benchmarks/results")
    p.add_argument("--dry-run", action="store_true")
    return p


def validate_args(a: argparse.Namespace) -> None:
    for name in ("requests", "concurrency", "input_lens", "output_lens"):
        if any(x <= 0 for x in getattr(a, name)):
            raise ValueError(f"{name.replace('_', '-')} values must be positive")
    if a.length_tolerance < 0 or a.warmup < 0 or a.repeats <= 0 or a.max_model_len <= 0 or a.timeout <= 0:
        raise ValueError("tolerance must be nonnegative; warmup nonnegative; repeats, max-model-len, timeout positive")
    if a.dataset != "synthetic" and not a.dataset_path:
        raise ValueError("--dataset-path is required for sharegpt/jsonl")
    if a.warmup > min(a.requests):
        raise ValueError("warmup cannot exceed the smallest request count")


def _git_metadata() -> dict[str, Any]:
    import subprocess
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:
        return {"git_commit": None, "git_dirty": None}


def write_outputs(args: argparse.Namespace, metadata: dict[str, Any], cases: list[dict[str, Any]], requests: list[dict[str, Any]], planned: dict[str, Any]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = Path(args.output_dir) / f"{stamp}-{os.getpid()}"
    out.mkdir(parents=True, exist_ok=False)
    summary = {"metadata": metadata, "planned": planned, "cases": cases}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = ["repeat", "input_target", "output_budget", "concurrency", "requested", "successes", "errors", "error_rate", "wall_seconds", "request_throughput_success_per_s", "estimated_input_tokens_per_s", "estimated_output_tokens_per_s", "input_tokens_min", "input_tokens_mean", "input_tokens_max", "workload_sha256", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "generated_tokens_mean", "generated_tokens_min", "generated_tokens_max", "ttft_ms", "tpot_ms", "itl_ms", "valid"]
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: c.get(k) for k in fields} for c in cases)
    if requests:
        with (out / "requests.jsonl").open("w", encoding="utf-8") as f:
            for row in requests:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        validate_args(args)
        tokenizer = load_tokenizer(args.tokenizer)
        records = load_records(Path(args.dataset_path), args.dataset) if args.dataset != "synthetic" else None
        if records is not None:
            for row in records:
                row["prompt"] = render_prompt(row["content"])
                row["input_tokens"] = len(encode_ids(tokenizer, row["prompt"], True))
                row["prompt_hash"] = sha256_bytes(row["prompt"].encode())
        dataset_hash = sha256_file(Path(args.dataset_path).expanduser()) if args.dataset_path else None
        tokenizer_path = Path(args.tokenizer).expanduser()
        tokenizer_hash = sha256_file(tokenizer_path) if tokenizer_path.is_file() else (sha256_file(tokenizer_path / "tokenizer.json") if (tokenizer_path / "tokenizer.json").is_file() else None)
        cases, request_rows = [], []
        manifest = []
        matrix = [(r, c, i, o) for r in args.requests for c in args.concurrency for i in args.input_lens for o in args.output_lens]
        cache: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
        for repeat in range(args.repeats):
            for count, concurrency, target, budget in matrix:
                key = (count, target, budget)
                if key not in cache:
                    stable_seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{count}:{target}".encode()).digest()[:8], "big")
                    cache[key] = prepare_workload(tokenizer, args.dataset, records, target, count, args.length_tolerance, stable_seed, budget, args.max_model_len)
                    manifest.append({"count": count, "target": target, "budget": budget, "prompt_hashes": [r["prompt_hash"] for r in cache[key]], "source_ids": [r["source_id"] for r in cache[key]], "input_token_lengths": [r["input_tokens"] for r in cache[key]], "record_indexes": [r.get("record_index") for r in cache[key]]})
                case_hash = workload_sha256(cache[key], budget)
                if args.dry_run:
                    lengths = [r["input_tokens"] for r in cache[key]]
                    cases.append({"repeat": repeat, "input_target": target, "output_budget": budget, "concurrency": concurrency, "requested": count, "input_tokens_min": min(lengths), "input_tokens_mean": mean(lengths), "input_tokens_max": max(lengths), "workload_sha256": case_hash, "valid": None, "ttft_ms": None, "tpot_ms": None, "itl_ms": None})
                    continue
                run, wall = run_requests(args.base_url, args.model, cache[key], budget, concurrency, args.timeout, tokenizer, args.warmup)
                for row in run:
                    row["workload_sha256"] = case_hash
                cases.append(summarize(run, wall, target, count, concurrency, budget, repeat, case_hash, [r["input_tokens"] for r in cache[key]]))
                request_rows.extend({k: v for k, v in row.items() if not k.startswith("_")} | {"repeat": repeat, "input_target": target, "output_budget": budget, "concurrency": concurrency} for row in run)
        metadata = {"status": "dry_run" if args.dry_run else "complete", "valid": None if args.dry_run else all(c["valid"] for c in cases), "model": args.model, "base_url": args.base_url, "argv": sys.argv, "seed": args.seed, "dataset": args.dataset, "dataset_sha256": dataset_hash, "tokenizer_sha256": tokenizer_hash, "label": args.label, "client_host": socket.gethostname(), "python": platform.python_version(), "git": _git_metadata(), "manifest_sha256": sha256_bytes(json.dumps(manifest, sort_keys=True).encode()), "no_server_token_counts": True, "timing_fields_unavailable": ["ttft_ms", "tpot_ms", "itl_ms"]}
        metadata["input_token_count_method"] = "rendered_prompt_with_special_tokens"
        metadata["output_token_count_method"] = "retokenized_text_estimate_without_special_tokens"
        metadata["load_pattern"] = "closed_loop_max_concurrency"
        metadata["benchmark_script_sha256"] = sha256_file(Path(__file__))
        planned = {"case_count": len(matrix) * args.repeats, "total_requests": sum(x[0] for x in matrix) * args.repeats, "manifest": manifest}
        out = write_outputs(args, metadata, cases, request_rows, planned)
        print(out)
        return 0 if args.dry_run or metadata["valid"] else 1
    except Exception as exc:
        print(f"benchmark error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
