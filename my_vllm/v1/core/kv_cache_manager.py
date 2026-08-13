"""
KV cache 抽象管理器 — KVCacheManager + KVCacheBlocks

对应 vLLM 的 vllm/v1/core/kv_cache_manager.py（大幅简化版）。

职责：封装 BlockPool（物理 block 池 + 前缀缓存），向 Scheduler 提供三个核心 API：
  ① get_computed_blocks() — 查前缀缓存命中，返回可复用的 block
  ② allocate_slots()      — 为请求分配新的 KV block
  ③ free()                — 释放请求占用的所有 block

Scheduler 不直接操作物理 block，所有 KV cache 账本逻辑由本类统一管理。

简化说明（相比 vLLM）：
  - 只支持单一 KV cache group（decoder-only 模型），对应 UnitaryKVCacheCoordinator
    的逻辑被拍平进本类，不再有 coordinator / single_type_manager 分层。
  - 前缀缓存 hash 由本类按 token id 现算，不再由 Request 提前算好 block_hashes。
"""

import logging
from dataclasses import dataclass
from typing import Sequence

from my_vllm.v1.core.kv_cache_utils import BlockHash, BlockPool, KVCacheBlock
from my_vllm.v1.request import Request

logger = logging.getLogger(__name__)


def cdiv(a: int, b: int) -> int:
    """向上取整除法：cdiv(10, 16) == 1"""
    return (a + b - 1) // b


@dataclass
class KVCacheBlocks:
    """分配结果 — Scheduler 与 KVCacheManager 之间的接口隔离层

    对应 vLLM 的 KVCacheBlocks。单 group 下 blocks 是长度为 1 的元组，
    blocks[0] 是一串 KVCacheBlock。Scheduler 只通过 get_block_ids() 拿 id 列表，
    看不到物理 block 的内部结构。
    """

    blocks: tuple[Sequence[KVCacheBlock], ...]

    def get_block_ids(self) -> tuple[list[int], ...]:
        """把 block 对象转成 block_id 列表"""
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        return KVCacheBlocks(
            tuple(list(b1) + list(b2) for b1, b2 in zip(self.blocks, other.blocks))
        )

    def new_empty(self) -> "KVCacheBlocks":
        return KVCacheBlocks(tuple(() for _ in self.blocks))


class KVCacheManager:
    """KV cache 管理器（简化版）

    Args:
        num_gpu_blocks: block 总数。
        block_size: 每个 block 容纳的 token 数。
        max_model_len: 模型最大上下文长度。
        enable_caching: 是否启用前缀缓存。
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        block_size: int,
        max_model_len: int,
        enable_caching: bool = True,
    ):
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        self.num_kv_cache_groups = 1  # decoder-only：单一 group

        self.block_pool = BlockPool(num_gpu_blocks, enable_caching)

        # 每个请求的 block 表：request_id -> [KVCacheBlock]（按 token 顺序）
        self.req_to_blocks: dict[str, list[KVCacheBlock]] = {}

        # 预构造的空 KVCacheBlocks，避免反复创建
        self.empty_kv_cache_blocks = KVCacheBlocks(((),))

    @property
    def usage(self) -> float:
        """KV cache 使用率（0.0 ~ 1.0）"""
        return self.block_pool.get_usage()

    # ---- 前缀缓存 ----

    @staticmethod
    def _get_block_hash(token_ids: Sequence[int]) -> BlockHash:
        """对一段 token id 计算 block 内容 hash"""
        return tuple(token_ids)

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """查前缀缓存命中，返回可复用的 block 与命中的 token 数

        对应 vLLM 的 KVCacheManager.get_computed_blocks()。

        注意：最多命中到 `num_tokens - 1`，因为最后一个 token 必须重新前向
        才能拿到它的 logits（采样需要）。因此命中长度按 block 对齐截断。
        """
        if not self.enable_caching or request.num_tokens <= 1:
            return self.empty_kv_cache_blocks, 0

        max_cache_hit_length = request.num_tokens - 1
        token_ids = request.all_token_ids

        hit_blocks: list[KVCacheBlock] = []
        for start in range(0, max_cache_hit_length, self.block_size):
            end = start + self.block_size
            if end > max_cache_hit_length:
                break  # 最后一个不满 block，不做整块命中
            block = self.block_pool.get_cached_block(
                self._get_block_hash(token_ids[start:end])
            )
            if block is None:
                break  # 前缀不再连续命中，停止
            hit_blocks.append(block)

        num_hit_tokens = len(hit_blocks) * self.block_size
        if hit_blocks:
            # touch：命中 block 的 ref_cnt +1（若原本是驱逐候选，从空闲队列移出）
            self.block_pool.touch(hit_blocks)
        return KVCacheBlocks((hit_blocks,)), num_hit_tokens

    # ---- 分配 ----

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
    ) -> KVCacheBlocks | None:
        """为请求分配新的 KV block，容纳 `num_new_tokens` 个新 token 的 KV

        对应 vLLM 的 KVCacheManager.allocate_slots()（简化版）。

        Args:
            request: 目标请求。
            num_new_tokens: 本轮要新增计算（写入 KV）的 token 数。
            num_new_computed_tokens: 前缀缓存刚命中的 token 数（首次调度时非 0）。
            new_computed_blocks: 命中的 block 列表，需挂载到请求的 block 表。

        Returns:
            新分配的 block（不含前缀命中的）；若显存不足返回 None。
        """
        if num_new_tokens <= 0:
            raise ValueError("num_new_tokens 必须大于 0")

        req_id = request.request_id
        blocks = list(self.req_to_blocks.get(req_id, []))

        # 1. 挂载前缀缓存命中的 block（首次调度时）
        if new_computed_blocks is not None and new_computed_blocks.blocks[0]:
            blocks.extend(new_computed_blocks.blocks[0])

        # 2. 计算需要新分配的 block 数
        # 已计算的 token = request 已有 computed + 本轮前缀命中
        total_computed = request.num_computed_tokens + num_new_computed_tokens
        total_need_slot = min(total_computed + num_new_tokens, self.max_model_len)

        num_blocks_needed = cdiv(total_need_slot, self.block_size)
        num_new_blocks = num_blocks_needed - len(blocks)

        # 3. 显存不足则失败（由 Scheduler 决定是否抢占）
        if num_new_blocks > 0 and num_new_blocks > self.block_pool.get_num_free_blocks():
            return None

        new_blocks: list[KVCacheBlock] = []
        if num_new_blocks > 0:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            blocks.extend(new_blocks)

        self.req_to_blocks[req_id] = blocks

        # 4. 把已写满的 block 注册进前缀缓存（供后续请求复用）
        num_tokens_to_cache = min(total_computed + num_new_tokens, request.num_tokens)
        self.cache_blocks(request, num_tokens_to_cache)

        return KVCacheBlocks((new_blocks,))

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """把请求前 num_computed_tokens 个 token 覆盖的「满 block」写进前缀缓存

        只有写满（block 对齐且已全部计算）的 block 才会被缓存。
        """
        if not self.enable_caching:
            return

        blocks = self.req_to_blocks.get(request.request_id, [])
        token_ids = request.all_token_ids
        num_full_blocks = num_computed_tokens // self.block_size

        for i in range(num_full_blocks):
            block = blocks[i]
            if block.block_hash is not None:
                continue  # 已缓存过
            start = i * self.block_size
            block_hash = self._get_block_hash(token_ids[start:start + self.block_size])
            block.set_block_hash(block_hash)
            self.block_pool.cache_block(block_hash, block)

    # ---- 查询 / 释放 ----

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """获取请求当前的完整 block 表"""
        return KVCacheBlocks((self.req_to_blocks.get(request_id, []),))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """获取请求的 block id 列表"""
        return self.get_blocks(request_id).get_block_ids()

    def free(self, request: Request) -> None:
        """释放请求占用的所有 block（引用计数 -1，归零回收）"""
        req_id = request.request_id
        blocks = self.req_to_blocks.pop(req_id, [])
        if blocks:
            # 按分配顺序传入，free_blocks 内部会让尾部 block 先成为驱逐候选
            self.block_pool.free_blocks(blocks)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """所有运行中请求的公共前缀 block 数（用于 cascade attention）

        简化：cascade attention 未实现，恒返回 0。
        """
        return [0]
