"""不依赖 transformers 的 Hugging Face tokenizer 适配层。"""

from __future__ import annotations

import json
from pathlib import Path


class HuggingFaceTokenizer:
    def __init__(self, model_path: str):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - 安装问题
            raise RuntimeError("真实在线推理需要安装 `pip install tokenizers`") from exc

        folder = Path(model_path).expanduser().resolve()
        tokenizer_path = folder / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"找不到 tokenizer.json: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.eos_token_ids = self._read_eos_token_ids(folder)

    @staticmethod
    def _read_eos_token_ids(folder: Path) -> tuple[int, ...]:
        for filename in ("generation_config.json", "config.json"):
            path = folder / filename
            if not path.is_file():
                continue
            with path.open(encoding="utf-8") as file:
                value = json.load(file).get("eos_token_id")
            if value is not None:
                values = value if isinstance(value, list) else [value]
                return tuple(int(token_id) for token_id in values)
        return ()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=True).ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

