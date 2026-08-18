"""Worker 侧持久化 block table。

Scheduler/BlockPool 管理 block 的所有权；这里仅保存每个在途请求收到的 block id
序列，供下一阶段计算 slot mapping 和 PagedAttention 读取。
"""

from __future__ import annotations


class MultiGroupBlockTable:
    """按 ``[group][req_index][logical_block_index]`` 保存 block id。"""

    def __init__(self, max_num_reqs: int, num_kv_cache_groups: int):
        if max_num_reqs <= 0:
            raise ValueError("max_num_reqs 必须大于 0")
        if num_kv_cache_groups <= 0:
            raise ValueError("num_kv_cache_groups 必须大于 0")
        self.max_num_reqs = max_num_reqs
        self.num_kv_cache_groups = num_kv_cache_groups
        self._rows = [
            [[] for _ in range(max_num_reqs)]
            for _ in range(num_kv_cache_groups)
        ]

    def _validate_groups(self, block_ids: tuple[list[int], ...]) -> None:
        if len(block_ids) != self.num_kv_cache_groups:
            raise ValueError(
                f"block table group 数不匹配：expected={self.num_kv_cache_groups}, "
                f"actual={len(block_ids)}"
            )

    def add_row(self, block_ids: tuple[list[int], ...], req_index: int) -> None:
        """写入完整 block table，用于新请求或抢占恢复。"""

        self._validate_groups(block_ids)
        for group_id, group_block_ids in enumerate(block_ids):
            self._rows[group_id][req_index] = list(group_block_ids)

    def append_row(
        self, new_block_ids: tuple[list[int], ...], req_index: int
    ) -> None:
        """追加本轮新分配的 block id，用于已缓存请求。"""

        self._validate_groups(new_block_ids)
        for group_id, group_block_ids in enumerate(new_block_ids):
            self._rows[group_id][req_index].extend(group_block_ids)

    def clear_row(self, req_index: int) -> None:
        for group_rows in self._rows:
            group_rows[req_index] = []

    def move_row(self, source_index: int, target_index: int) -> None:
        for group_rows in self._rows:
            group_rows[target_index] = group_rows[source_index]
            group_rows[source_index] = []

    def get_row(self, req_index: int) -> tuple[list[int], ...]:
        return tuple(
            list(group_rows[req_index]) for group_rows in self._rows
        )

    def __getitem__(self, group_id: int) -> list[list[int]]:
        return self._rows[group_id]
