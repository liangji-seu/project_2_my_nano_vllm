"""生成可复用的 Qwen2.5-7B 固定压测请求集。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer


ROUTE_PREFIX = "[user]: "
FILLERS = (
    " benchmark",
    " token",
    " inference",
    " data",
    " model",
    " a",
    " 0",
    ".",
    "测",
)


def _make_exact_prompt(
    tokenizer: Tokenizer,
    request_index: int,
    target_tokens: int,
) -> str:
    """构造经在线路由拼接后恰好为 target_tokens 的稳定文本。"""

    content = (
        f"Fixed benchmark request {request_index:04d}. "
        "Analyze deterministic language-model serving workload."
    )

    def count(candidate: str) -> int:
        return len(
            tokenizer.encode(
                ROUTE_PREFIX + candidate, add_special_tokens=True
            ).ids
        )

    # 大段填充用二分搜索，避免对逐渐增长的千Token字符串做上千次重编码。
    unit = " benchmark"
    low, high = 0, target_tokens
    while low < high:
        middle = (low + high + 1) // 2
        if count(content + unit * middle) <= target_tokens:
            low = middle
        else:
            high = middle - 1
    content += unit * low

    # BPE边界可能让最后还差少量token，再用多种短片段精确补齐。
    while count(content) < target_tokens:
        current = count(content)
        best: str | None = None
        best_count = current
        for filler in FILLERS:
            candidate = content + filler
            candidate_count = count(candidate)
            if best_count < candidate_count <= target_tokens:
                best = candidate
                best_count = candidate_count
        if best is None:
            raise RuntimeError(
                f"无法把 request={request_index} 精确填充到 {target_tokens} tokens，"
                f"当前为 {current}"
            )
        content = best

    actual = count(content)
    if actual != target_tokens:
        raise AssertionError(f"prompt token数错误: {actual} != {target_tokens}")
    return content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-requests", type=int, default=256)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with args.output.open("w", encoding="utf-8") as file:
        for index in range(args.num_requests):
            content = _make_exact_prompt(tokenizer, index, args.prompt_tokens)
            record = {
                "request_id": f"qwen2.5-7b-baseline-{index:04d}",
                "messages": [{"role": "user", "content": content}],
                "prompt_tokens": args.prompt_tokens,
                "max_tokens": args.max_tokens,
                "max_model_len": args.max_model_len,
                "ignore_eos": True,
            }
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            digest.update((line + "\n").encode())
            file.write(line + "\n")

    print(
        json.dumps(
            {
                "output": str(args.output),
                "num_requests": args.num_requests,
                "prompt_tokens": args.prompt_tokens,
                "max_tokens": args.max_tokens,
                "max_model_len": args.max_model_len,
                "sha256": digest.hexdigest(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
