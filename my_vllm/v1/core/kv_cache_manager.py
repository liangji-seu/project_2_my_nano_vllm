"""Scheduler 面向的 KV cache 总入口。

对象关系与 vLLM 保持一致：

``KVCacheManager -> KVCacheCoordinator -> SingleTypeKVCacheManager``
``                                      -> shared BlockPool``

KVCacheManager 只负责编排，不直接维护某个请求的 block table。
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from my_vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    KVCacheCoordinator,
    UnitaryKVCacheCoordinator,
)
from my_vllm.v1.core.kv_cache_utils import KVCacheBlock
from my_vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    make_default_kv_cache_config,
)
from my_vllm.v1.request import Request


@dataclass(frozen=True)
class KVCacheBlocks:
    """多 group block table 的只读传输包装。"""

    blocks: tuple[Sequence[KVCacheBlock], ...]

    def get_block_ids(self) -> tuple[list[int], ...]:
        return tuple(
            [block.block_id for block in group_blocks]
            for group_blocks in self.blocks
        )

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        if len(self.blocks) != len(other.blocks):
            raise ValueError("KV cache group 数量不一致")
        return KVCacheBlocks(
            tuple(
                list(left) + list(right)
                for left, right in zip(self.blocks, other.blocks)
            )
        )

    def new_empty(self) -> "KVCacheBlocks":
        return KVCacheBlocks(tuple(() for _ in self.blocks))


class KVCacheManager:
    """KV cache CPU 侧逻辑账本，对 Scheduler 提供稳定 API。

    旧的 ``num_gpu_blocks + block_size`` 构造方式仍然可用；内部会先转换成
    单 group ``KVCacheConfig``。当 Worker profiling 接入后，可直接传完整配置。
    """

    def __init__(
        self,
        num_gpu_blocks: int | None = None,
        block_size: int | None = None,
        max_model_len: int = 4096,
        enable_caching: bool = True,
        *,
        kv_cache_config: KVCacheConfig | None = None,
        watermark_blocks: int = 0,
    ) -> None:
        if kv_cache_config is None:
            if num_gpu_blocks is None or block_size is None:
                raise ValueError(
                    "必须传 kv_cache_config，或同时传 num_gpu_blocks/block_size"
                )
            kv_cache_config = make_default_kv_cache_config(
                num_blocks=num_gpu_blocks,
                block_size=block_size,
            )

        group_block_sizes = [
            group.kv_cache_spec.block_size
            for group in kv_cache_config.kv_cache_groups
        ]
        self.scheduler_block_size = math.lcm(*group_block_sizes)
        self.hash_block_size = math.gcd(*group_block_sizes)
        self.block_size = self.scheduler_block_size
        self.max_model_len = max_model_len
        self.max_in_flight_tokens = max_model_len
        self.enable_caching = enable_caching
        self.kv_cache_config = kv_cache_config
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.watermark_blocks = watermark_blocks

        coordinator_cls: type[KVCacheCoordinator]
        if self.num_kv_cache_groups == 1:
            coordinator_cls = UnitaryKVCacheCoordinator
        else:
            coordinator_cls = HybridKVCacheCoordinator
        self.coordinator = coordinator_cls(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            enable_caching=enable_caching,
            scheduler_block_size=self.scheduler_block_size,
            hash_block_size=self.hash_block_size,
        )
        self.block_pool = self.coordinator.block_pool
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

    @property
    def usage(self) -> float:
        return self.block_pool.get_usage()

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """查所有 group 共同可复用的最长连续前缀。"""

        if not self.enable_caching or request.num_tokens <= 1:
            return self.empty_kv_cache_blocks, 0
        request.update_block_hashes(self.hash_block_size)
        max_cache_hit_length = request.num_tokens - 1
        blocks, num_computed_tokens = self.coordinator.find_longest_cache_hit(
            request.block_hashes,
            max_cache_hit_length,
        )
        return self.create_kv_cache_blocks(blocks), num_computed_tokens

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
    ) -> KVCacheBlocks | None:
        """挂载命中块并为本轮 token 分配新 block；此处绝不登记缓存。"""

        if num_new_tokens <= 0:
            raise ValueError("num_new_tokens 必须大于 0")
        if new_computed_blocks is None:
            new_computed_blocks = self.empty_kv_cache_blocks
        if len(new_computed_blocks.blocks) != self.num_kv_cache_groups:
            raise ValueError("new_computed_blocks 的 group 数量不匹配")

        total_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_tokens = min(
            total_computed_tokens + num_new_tokens,
            self.max_model_len,
        )
        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request.request_id,
            total_tokens,
            new_computed_blocks.blocks,
        )
        if (
            num_blocks_to_allocate + self.watermark_blocks
            > self.block_pool.get_num_free_blocks()
        ):
            return None

        self.coordinator.allocate_new_computed_blocks(
            request.request_id,
            new_computed_blocks.blocks,
        )
        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            total_tokens,
        )
        return self.create_kv_cache_blocks(new_blocks)

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """模型执行完成后，登记已真正写好的 full blocks。"""

        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        return self.get_blocks(request_id).get_block_ids()

    def free(self, request: Request) -> None:
        self.coordinator.free(request.request_id)

    def get_num_common_prefix_blocks(self, request_id: str) -> list[int]:
        if not request_id:
            return [0] * self.num_kv_cache_groups
        return self.coordinator.get_num_common_prefix_blocks(request_id)

    def reset_prefix_cache(self) -> bool:
        return self.block_pool.reset_prefix_cache()

    def create_kv_cache_blocks(
        self,
        blocks: tuple[Sequence[KVCacheBlock], ...],
    ) -> KVCacheBlocks:
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks
