"""简化 FullAttention 前向所需的运行时元数据。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FullAttentionMetadata:
    """一个 KV cache group 共享的 FullAttention 运行时描述。

    ``slot_mapping`` 服务当前 token 的 K/V 写入；``block_table`` 和
    ``seq_lens`` 服务 Attention 对历史 K/V 的分页读取。这里不引入 backend
    builder，后续 FullAttention 算子可以直接消费这个对象。
    """

    kv_cache_group_id: int
    layer_names: tuple[str, ...]
    block_size: int
    causal: bool

    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int

    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    num_computed_tokens: torch.Tensor
    num_scheduled_tokens: torch.Tensor
    positions: torch.Tensor

    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    def __post_init__(self) -> None:
        if self.query_start_loc.shape != (self.num_reqs + 1,):
            raise ValueError("query_start_loc 形状必须是 [num_reqs + 1]")
        for name, tensor in (
            ("seq_lens", self.seq_lens),
            ("num_computed_tokens", self.num_computed_tokens),
            ("num_scheduled_tokens", self.num_scheduled_tokens),
        ):
            if tensor.shape != (self.num_reqs,):
                raise ValueError(f"{name} 形状必须是 [num_reqs]")
        if self.positions.shape != (self.num_actual_tokens,):
            raise ValueError("positions 形状必须是 [num_actual_tokens]")
        if self.slot_mapping.shape != (self.num_actual_tokens,):
            raise ValueError("slot_mapping 形状必须是 [num_actual_tokens]")
        if self.block_table.ndim != 2 or self.block_table.shape[0] != self.num_reqs:
            raise ValueError("block_table 形状必须是 [num_reqs, max_num_blocks]")

    @property
    def query_lens(self) -> torch.Tensor:
        """每个请求本轮 Q 的长度。"""

        return self.num_scheduled_tokens

    @property
    def context_lens(self) -> torch.Tensor:
        """进入本轮前已经存在的历史 KV 长度。"""

        return self.num_computed_tokens


@dataclass(frozen=True)
class FullAttentionMetadataCollection:
    """同时提供 group 视图和 layer 视图的 metadata 容器。"""

    by_group: dict[int, FullAttentionMetadata]
    by_layer: dict[str, FullAttentionMetadata]

    def for_layer(self, layer_name: str) -> FullAttentionMetadata:
        try:
            return self.by_layer[layer_name]
        except KeyError as exc:
            raise KeyError(f"没有 layer={layer_name!r} 的 Attention metadata") from exc
