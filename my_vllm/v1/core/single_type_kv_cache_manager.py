"""单个 KV cache group 的 block table 管理器。"""

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence

from my_vllm.v1.core.block_pool import BlockPool
from my_vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from my_vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec, MambaSpec
from my_vllm.v1.request import Request


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


class SingleTypeKVCacheManager(ABC):
    """管理一种 KV cache 形状下，每个请求自己的 block table。"""

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        scheduler_block_size: int,
    ) -> None:
        if scheduler_block_size % kv_cache_spec.block_size != 0:
            raise ValueError("scheduler_block_size 必须是 group block_size 的倍数")
        self.scheduler_block_size = scheduler_block_size
        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.enable_caching = enable_caching
        self.kv_cache_group_id = kv_cache_group_id

        # 每个 group 分别维护 req -> block table；Request 自身不保存 block id。
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)
        self.num_cached_block: dict[str, int] = {}

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
    ) -> int:
        num_required_blocks = cdiv(num_tokens, self.block_size)
        current_blocks = len(self.req_to_blocks.get(request_id, ()))

        if request_id in self.num_cached_block:
            if new_computed_blocks:
                raise RuntimeError("running 请求不应再次挂载 Prefix Cache 命中块")
            return max(num_required_blocks - current_blocks, 0)

        num_new = max(
            num_required_blocks - current_blocks - len(new_computed_blocks),
            0,
        )
        # ref_cnt=0 的命中块仍在 free queue 中，touch 后也会消耗空闲容量。
        evictable_hits = sum(block.ref_cnt == 0 for block in new_computed_blocks)
        return num_new + evictable_hits

    def add_local_computed_blocks(
        self,
        request_id: str,
        computed_blocks: Sequence[KVCacheBlock],
    ) -> None:
        if request_id in self.num_cached_block:
            if computed_blocks:
                raise RuntimeError("running 请求不能追加新的 Prefix Cache 命中块")
            return
        blocks = list(computed_blocks)
        if blocks:
            self.block_pool.touch(blocks)
            self.req_to_blocks[request_id].extend(blocks)
        self.num_cached_block[request_id] = len(blocks)

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
    ) -> list[KVCacheBlock]:
        request_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = max(num_required_blocks - len(request_blocks), 0)
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        request_blocks.extend(new_blocks)
        return new_blocks

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        """把已经执行完成的 full blocks 登记到 Prefix Cache。"""

        if not self.enable_caching:
            return
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size
        if num_cached_blocks >= num_full_blocks:
            return
        blocks = self.req_to_blocks[request.request_id]
        self.block_pool.cache_full_blocks(
            request=request,
            blocks=blocks,
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size,
            kv_cache_group_id=self.kv_cache_group_id,
        )
        self.num_cached_block[request.request_id] = num_full_blocks

    def get_blocks(self, request_id: str) -> list[KVCacheBlock]:
        return self.req_to_blocks.get(request_id, [])

    def free(self, request_id: str) -> None:
        blocks = self.req_to_blocks.pop(request_id, [])
        self.num_cached_block.pop(request_id, None)
        self.block_pool.free_blocks(reversed(blocks))

    @classmethod
    @abstractmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        raise NotImplementedError

    @abstractmethod
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        raise NotImplementedError


class FullAttentionManager(SingleTypeKVCacheManager):
    """全注意力 group：请求结束前保留完整的 KV block table。"""

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        block_size = kv_cache_spec.block_size
        computed: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in kv_cache_group_ids
        )
        max_blocks = max_length // block_size
        hash_scale = block_size // block_pool.hash_block_size

        for block_index in range(max_blocks):
            hash_index = (block_index + 1) * hash_scale - 1
            cached = block_pool.get_cached_block(
                block_hashes[hash_index],
                kv_cache_group_ids,
            )
            if cached is None:
                break
            for group_blocks, block in zip(computed, cached):
                group_blocks.append(block)
        return computed, len(computed[0]) * block_size

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        blocks = self.req_to_blocks.get(running_request_id, [])
        num_requests = len(self.req_to_blocks)
        common = 0
        for block in blocks:
            if block.ref_cnt != num_requests:
                break
            common += 1
        return common


def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec,
    block_pool: BlockPool,
    enable_caching: bool,
    kv_cache_group_id: int,
    scheduler_block_size: int,
) -> SingleTypeKVCacheManager:
    """按照 group spec 选择对应的 block table 管理器。"""

    if isinstance(kv_cache_spec, FullAttentionSpec):
        return FullAttentionManager(
            kv_cache_spec=kv_cache_spec,
            block_pool=block_pool,
            enable_caching=enable_caching,
            kv_cache_group_id=kv_cache_group_id,
            scheduler_block_size=scheduler_block_size,
        )
    if isinstance(kv_cache_spec, MambaSpec):
        raise NotImplementedError("Mamba group 已有配置类型，但管理策略尚未实现")
    raise TypeError(f"不支持的 KV cache spec: {type(kv_cache_spec).__name__}")
