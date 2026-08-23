"""仅针对纯 Decode 的整模型 CUDA Graph 捕获与回放。

这里刻意不实现 piecewise graph：Prefill 和混合 batch 继续 eager，只有每个
request 本轮都恰好计算一个 token、且已经存在历史 KV 时，才把完整 Qwen2
forward 录入一张 CUDA Graph。采样和 D2H 仍留在图外。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from my_vllm.worker.gpu_model_runner import ModelForwardInputs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecodeCUDAGraphKey:
    """一张 FULL Decode 图的静态启动规格。

    Decode 下 ``num_tokens == num_reqs``。二者仍同时保留在 key 中，让这个
    约束在调试输出里一眼可见。``max_seq_len`` 是已经向上取整后的 Attention
    循环上界；真实长度继续由固定地址的 ``seq_lens`` tensor 提供。
    """

    num_tokens: int
    num_reqs: int
    max_seq_len: int


@dataclass
class _CUDAGraphEntry:
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    input_addresses: tuple[int, ...]


class FullDecodeCUDAGraphRunner:
    """在首次遇到某个 Decode 规格时惰性捕获，之后直接 replay。

    外部 ``GPUModelRunner`` 负责把每轮数据覆写到固定 GPU buffer。本类不复制
    输入，只验证地址没有变化。这与 vLLM 的职责边界一致：wrapper 只负责捕获
    和回放，静态输入缓冲由 model runner 管理。
    """

    def __init__(
        self,
        runnable: Callable[["ModelForwardInputs"], torch.Tensor],
        *,
        device: torch.device,
        seq_len_bucket_size: int,
        num_warmups: int = 1,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("FullDecodeCUDAGraphRunner 只支持 CUDA device")
        if seq_len_bucket_size <= 0:
            raise ValueError("seq_len_bucket_size 必须大于 0")
        if num_warmups < 0:
            raise ValueError("num_warmups 不能为负数")
        self.runnable = runnable
        self.device = device
        self.seq_len_bucket_size = seq_len_bucket_size
        self.num_warmups = num_warmups
        self.graph_pool = torch.cuda.graph_pool_handle()
        self.entries: dict[DecodeCUDAGraphKey, _CUDAGraphEntry] = {}
        self.capture_count = 0
        self.replay_count = 0

    def bucket_seq_len(self, seq_len: int) -> int:
        """把动态历史长度变成少量可复用的静态 Attention 循环上界。"""

        if seq_len <= 0:
            raise ValueError("seq_len 必须大于 0")
        bucket = self.seq_len_bucket_size
        return ((seq_len + bucket - 1) // bucket) * bucket

    @staticmethod
    def can_run(model_inputs: "ModelForwardInputs") -> bool:
        """FULL 图目前只接受已经有历史 KV 的非投机纯 Decode batch。"""

        if not model_inputs.is_decode:
            return False
        metadata_values = tuple(model_inputs.attention_metadata.by_group.values())
        if not metadata_values:
            return False
        return all(
            metadata.max_query_len == 1
            and metadata.num_actual_tokens == metadata.num_reqs
            for metadata in metadata_values
        )

    @staticmethod
    def _key(model_inputs: "ModelForwardInputs") -> DecodeCUDAGraphKey:
        metadata = next(iter(model_inputs.attention_metadata.by_group.values()))
        return DecodeCUDAGraphKey(
            num_tokens=metadata.num_actual_tokens,
            num_reqs=metadata.num_reqs,
            max_seq_len=metadata.max_seq_len,
        )

    @staticmethod
    def _input_addresses(model_inputs: "ModelForwardInputs") -> tuple[int, ...]:
        """收集真正进入整模型图的固定 Tensor 地址，排除图外 logits_indices。"""

        addresses = [
            model_inputs.input_ids.data_ptr(),
            model_inputs.positions.data_ptr(),
        ]
        for group_id in sorted(model_inputs.attention_metadata.by_group):
            metadata = model_inputs.attention_metadata.by_group[group_id]
            addresses.extend(
                tensor.data_ptr()
                for tensor in (
                    metadata.query_start_loc,
                    metadata.seq_lens,
                    metadata.num_computed_tokens,
                    metadata.num_scheduled_tokens,
                    metadata.positions,
                    metadata.block_table,
                    metadata.slot_mapping,
                )
            )
        return tuple(addresses)

    @torch.inference_mode()
    def __call__(self, model_inputs: "ModelForwardInputs") -> torch.Tensor:
        if not self.can_run(model_inputs):
            return self.runnable(model_inputs)

        key = self._key(model_inputs)
        current_addresses = self._input_addresses(model_inputs)
        entry = self.entries.get(key)
        if entry is not None:
            if current_addresses != entry.input_addresses:
                raise RuntimeError(
                    "CUDA Graph replay 输入地址发生变化："
                    f"key={key}, expected={entry.input_addresses}, "
                    f"actual={current_addresses}"
                )
            entry.graph.replay()
            self.replay_count += 1
            return entry.output

        entry = self._capture(key, model_inputs, current_addresses)
        self.entries[key] = entry
        self.capture_count += 1
        logger.info(
            "【FULL CUDA Graph】捕获纯 Decode 图：num_reqs=%d, "
            "num_tokens=%d, max_seq_len_bucket=%d",
            key.num_reqs,
            key.num_tokens,
            key.max_seq_len,
        )
        return entry.output

    def _capture(
        self,
        key: DecodeCUDAGraphKey,
        model_inputs: "ModelForwardInputs",
        input_addresses: tuple[int, ...],
    ) -> _CUDAGraphEntry:
        """先在旁路 stream warmup，再捕获完整 model forward。"""

        current_stream = torch.cuda.current_stream(self.device)
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(self.num_warmups):
                warmup_output = self.runnable(model_inputs)
            if self.num_warmups:
                del warmup_output
        current_stream.wait_stream(warmup_stream)
        # 首次捕获是启动阶段的一次性同步；正常 replay 路径没有同步点。
        torch.cuda.synchronize(self.device)

        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.graph(
            graph,
            pool=self.graph_pool,
            stream=capture_stream,
        ):
            output = self.runnable(model_inputs)
        current_stream.wait_stream(capture_stream)
        torch.cuda.synchronize(self.device)
        logger.debug("CUDA Graph capture完成：%s", key)
        return _CUDAGraphEntry(
            graph=graph,
            output=output,
            input_addresses=input_addresses,
        )

    def clear(self) -> None:
        self.entries.clear()
