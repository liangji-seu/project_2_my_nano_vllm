"""Worker 侧持久化请求状态与 InputBatch。

这一层只维护跨 step 的 CPU 状态；``_prepare_inputs`` 所需的扁平 input_ids、
positions、query_start_loc、slot_mapping 等 GPU 输入留给下一阶段实现。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from my_vllm.v1.request import SamplingParams
from my_vllm.worker.block_table import MultiGroupBlockTable


@dataclass
class CachedRequestState:
    """GPUModelRunner 为一个请求保存的完整、跨 step 状态。"""

    req_id: str
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    sampling_params: SamplingParams
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids


class InputBatch:
    """用稳定的 ``req_index`` 槽位维护当前 Worker 的在途 batch。

    请求的完整状态同时保存在 ``GPUModelRunner.requests``；InputBatch 是适合
    批量整理模型输入的槽位化视图。请求本轮未调度时会移出这里，但缓存状态保留。
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        num_kv_cache_groups: int,
        pin_memory: bool = False,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len

        self._req_ids: list[str | None] = []
        self.req_id_to_index: dict[str, int] = {}

        # 跨 step 持久化的二维 CPU token 表；ModelRunner 每轮用 token_indices
        # 从中抽取本轮 query，整理成扁平 input_ids。
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()
        # 当前请求协议只有 token ids，没有 prompt embeds。仍保留该掩码，使后续
        # 接入 embeddings 时无需改变 _prepare_inputs 的数据模型。
        self.is_token_ids_tensor = torch.ones(
            (max_num_reqs, max_model_len),
            dtype=torch.bool,
            pin_memory=pin_memory,
        )
        self.req_output_token_ids: list[list[int] | None] = []
        self.num_tokens_no_spec_tensor = torch.zeros(
            max_num_reqs, dtype=torch.int32, pin_memory=pin_memory
        )
        self.num_tokens_no_spec = self.num_tokens_no_spec_tensor.numpy()
        self.num_prompt_tokens_tensor = torch.zeros(
            max_num_reqs, dtype=torch.int32, pin_memory=pin_memory
        )
        self.num_prompt_tokens = self.num_prompt_tokens_tensor.numpy()
        self.num_computed_tokens_cpu_tensor = torch.zeros(
            max_num_reqs, dtype=torch.int32, pin_memory=pin_memory
        )
        self.num_computed_tokens_cpu = self.num_computed_tokens_cpu_tensor.numpy()
        # baseline 没有投机解码：每轮有效输出数固定为 1。
        self.num_accepted_tokens_cpu_tensor = torch.ones(
            max_num_reqs, dtype=torch.int32, pin_memory=pin_memory
        )
        self.num_accepted_tokens_cpu = self.num_accepted_tokens_cpu_tensor.numpy()

        self.temperature_tensor = torch.zeros(
            max_num_reqs, dtype=torch.float32, pin_memory=pin_memory
        )
        self.temperature = self.temperature_tensor.numpy()
        self.top_p_tensor = torch.ones(
            max_num_reqs, dtype=torch.float32, pin_memory=pin_memory
        )
        self.top_p = self.top_p_tensor.numpy()

        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            num_kv_cache_groups=num_kv_cache_groups,
        )

    @property
    def req_ids(self) -> list[str]:
        return [req_id for req_id in self._req_ids if req_id is not None]

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    def add_request(self, request: CachedRequestState) -> int:
        if request.req_id in self.req_id_to_index:
            raise ValueError(f"请求 {request.req_id} 已经在 InputBatch 中")
        if request.num_tokens > self.max_model_len:
            raise ValueError(
                f"请求 {request.req_id} token 数 {request.num_tokens} "
                f"超过 max_model_len={self.max_model_len}"
            )
        if self.num_reqs >= self.max_num_reqs:
            raise RuntimeError("InputBatch 已达到 max_num_reqs")

        try:
            req_index = self._req_ids.index(None)
        except ValueError:
            req_index = len(self._req_ids)
            self._req_ids.append(request.req_id)
            self.req_output_token_ids.append(request.output_token_ids)
        else:
            self._req_ids[req_index] = request.req_id
            self.req_output_token_ids[req_index] = request.output_token_ids

        self.req_id_to_index[request.req_id] = req_index
        token_ids = request.all_token_ids
        self.token_ids_cpu[req_index][: len(token_ids)] = token_ids
        self.is_token_ids_tensor[req_index, : len(token_ids)] = True
        self.num_tokens_no_spec[req_index] = request.num_tokens
        self.num_prompt_tokens[req_index] = request.num_prompt_tokens
        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        self.num_accepted_tokens_cpu[req_index] = 1
        self.temperature[req_index] = request.sampling_params.temperature
        self.top_p[req_index] = request.sampling_params.top_p
        self.block_table.add_row(request.block_ids, req_index)
        return req_index

    def remove_request(self, req_id: str) -> int | None:
        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None
        self._req_ids[req_index] = None
        self.req_output_token_ids[req_index] = None
        self.num_tokens_no_spec[req_index] = 0
        self.num_prompt_tokens[req_index] = 0
        self.num_computed_tokens_cpu[req_index] = 0
        self.num_accepted_tokens_cpu[req_index] = 1
        self.is_token_ids_tensor[req_index].fill_(True)
        self.block_table.clear_row(req_index)
        return req_index

    def append_output_token_ids(self, req_id: str, token_ids: list[int]) -> None:
        if not token_ids:
            return
        req_index = self.req_id_to_index[req_id]
        start = self.num_tokens_no_spec[req_index]
        end = start + len(token_ids)
        if end > self.max_model_len:
            raise ValueError(f"请求 {req_id} 追加 token 后超过 max_model_len")
        self.token_ids_cpu[req_index][start:end] = token_ids
        self.num_tokens_no_spec[req_index] = end

    def condense(self) -> None:
        """把尾部活跃槽位搬进前面的空洞，保持 batch 紧凑。"""

        empty_indices = [
            index for index, req_id in enumerate(self._req_ids) if req_id is None
        ]
        while empty_indices:
            empty_index = empty_indices[0]
            last_index = len(self._req_ids) - 1
            while last_index >= 0 and self._req_ids[last_index] is None:
                self._req_ids.pop()
                self.req_output_token_ids.pop()
                last_index -= 1
            if empty_index >= len(self._req_ids):
                break

            req_id = self._req_ids[last_index]
            assert req_id is not None
            num_tokens = self.num_tokens_no_spec[last_index]

            self._req_ids[empty_index] = req_id
            self._req_ids.pop()
            self.req_output_token_ids[empty_index] = self.req_output_token_ids[last_index]
            self.req_output_token_ids.pop()
            self.req_id_to_index[req_id] = empty_index

            self.token_ids_cpu[empty_index][:num_tokens] = self.token_ids_cpu[
                last_index
            ][:num_tokens]
            self.is_token_ids_tensor[empty_index, :num_tokens].copy_(
                self.is_token_ids_tensor[last_index, :num_tokens]
            )
            self.is_token_ids_tensor[last_index].fill_(True)
            self.num_tokens_no_spec[empty_index] = num_tokens
            self.num_prompt_tokens[empty_index] = self.num_prompt_tokens[last_index]
            self.num_computed_tokens_cpu[empty_index] = (
                self.num_computed_tokens_cpu[last_index]
            )
            self.num_accepted_tokens_cpu[empty_index] = (
                self.num_accepted_tokens_cpu[last_index]
            )
            self.temperature[empty_index] = self.temperature[last_index]
            self.top_p[empty_index] = self.top_p[last_index]
            self.block_table.move_row(last_index, empty_index)

            empty_indices = [
                index
                for index, current_req_id in enumerate(self._req_ids)
                if current_req_id is None
            ]

    def snapshot(self, req_id: str) -> dict:
        """测试/学习用快照，直观看到一个槽位的三类核心状态。"""

        req_index = self.req_id_to_index[req_id]
        num_tokens = self.num_tokens_no_spec[req_index]
        return {
            "req_index": req_index,
            "token_ids": list(self.token_ids_cpu[req_index][:num_tokens]),
            "num_prompt_tokens": self.num_prompt_tokens[req_index],
            "num_tokens": num_tokens,
            "num_computed_tokens": self.num_computed_tokens_cpu[req_index],
            "block_ids": self.block_table.get_row(req_index),
        }
