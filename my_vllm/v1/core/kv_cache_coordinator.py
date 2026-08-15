"""协调多个 KV cache group，并让它们共享同一个 BlockPool。"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from my_vllm.v1.core.block_pool import BlockPool
from my_vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from my_vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SingleTypeKVCacheManager,
    get_manager_for_kv_cache_spec,
)
from my_vllm.v1.kv_cache_interface import KVCacheConfig
from my_vllm.v1.request import Request


class KVCacheCoordinator(ABC):
    """KV cache group 的统一协调层。

    Coordinator 持有一个共享 BlockPool；每个 group manager 只维护本 group
    下的请求 block table，并通过同一个池申请抽象 block id。
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool,
        scheduler_block_size: int,
        hash_block_size: int,
    ) -> None:
        if scheduler_block_size % hash_block_size != 0:
            raise ValueError("scheduler_block_size 必须是 hash_block_size 的倍数")
        for group in kv_cache_config.kv_cache_groups:
            if scheduler_block_size % group.kv_cache_spec.block_size != 0:
                raise ValueError("调度粒度必须对齐所有 KV cache group")

        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        self.scheduler_block_size = scheduler_block_size
        self.block_pool = BlockPool(
            num_gpu_blocks=kv_cache_config.num_blocks,
            enable_caching=enable_caching,
            hash_block_size=hash_block_size,
        )
        self.single_type_managers: tuple[SingleTypeKVCacheManager, ...] = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=group.kv_cache_spec,
                block_pool=self.block_pool,
                enable_caching=enable_caching,
                kv_cache_group_id=group_id,
                scheduler_block_size=scheduler_block_size,
            )
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        )

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
    ) -> int:
        return sum(
            manager.get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks[group_id],
            )
            for group_id, manager in enumerate(self.single_type_managers)
        )

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
    ) -> None:
        # 先 touch 所有 group 的命中块，再分配新块，避免前一 group 的分配
        # 驱逐后一 group 尚未 touch 的 ref_cnt=0 命中块。
        for group_id, manager in enumerate(self.single_type_managers):
            manager.add_local_computed_blocks(
                request_id,
                new_computed_blocks[group_id],
            )

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
    ) -> tuple[list[KVCacheBlock], ...]:
        return tuple(
            manager.allocate_new_blocks(request_id, num_tokens)
            for manager in self.single_type_managers
        )

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        for manager in self.single_type_managers:
            manager.cache_blocks(request, num_computed_tokens)

    def free(self, request_id: str) -> None:
        for manager in self.single_type_managers:
            manager.free(request_id)

    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        return tuple(
            manager.get_blocks(request_id)
            for manager in self.single_type_managers
        )

    def get_num_common_prefix_blocks(self, request_id: str) -> list[int]:
        return [
            manager.get_num_common_prefix_blocks(request_id)
            for manager in self.single_type_managers
        ]

    @abstractmethod
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        raise NotImplementedError


class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    """普通 decoder-only 模型使用的单 KV cache group 协调器。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if len(self.single_type_managers) != 1:
            raise ValueError("UnitaryKVCacheCoordinator 只支持一个 group")

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        manager = self.single_type_managers[0]
        return manager.find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.block_pool,
            kv_cache_spec=manager.kv_cache_spec,
        )


class HybridKVCacheCoordinator(KVCacheCoordinator):
    """多 group 协调器；当前实现全注意力 group 的共同最长命中。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if len(self.single_type_managers) < 2:
            raise ValueError("HybridKVCacheCoordinator 至少需要两个 group")
        if not all(
            isinstance(manager, FullAttentionManager)
            for manager in self.single_type_managers
        ):
            raise NotImplementedError("当前 Hybrid 只实现 FullAttention group")

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """按 scheduler 对齐粒度查所有 group 共同可用的最长前缀。"""

        hit_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in self.single_type_managers
        )
        hit_length = 0
        max_aligned = (
            max_cache_hit_length // self.scheduler_block_size
        ) * self.scheduler_block_size

        for boundary in range(
            self.scheduler_block_size,
            max_aligned + 1,
            self.scheduler_block_size,
        ):
            additions: list[list[KVCacheBlock]] = []
            all_groups_hit = True
            for group_id, manager in enumerate(self.single_type_managers):
                previous_blocks = hit_length // manager.block_size
                boundary_blocks = boundary // manager.block_size
                group_additions: list[KVCacheBlock] = []
                for block_index in range(previous_blocks, boundary_blocks):
                    prefix_tokens = (block_index + 1) * manager.block_size
                    hash_index = prefix_tokens // self.block_pool.hash_block_size - 1
                    cached = self.block_pool.get_cached_block(
                        block_hashes[hash_index],
                        [group_id],
                    )
                    if cached is None:
                        all_groups_hit = False
                        break
                    group_additions.append(cached[0])
                additions.append(group_additions)
                if not all_groups_hit:
                    break
            if not all_groups_hit:
                break
            for group_blocks, group_additions in zip(hit_blocks, additions):
                group_blocks.extend(group_additions)
            hit_length = boundary

        return hit_blocks, hit_length
