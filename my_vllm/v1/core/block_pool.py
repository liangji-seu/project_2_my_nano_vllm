"""所有 KV cache group 共享的抽象 BlockPool。"""

from collections.abc import Iterable

from my_vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from my_vllm.v1.request import Request


class BlockHashToBlockMap:
    """Prefix Cache 的 hash -> block 正向索引。

    同一内容可能在并发请求中被重复计算，所以一个 hash 允许对应多个 block。
    """

    def __init__(self) -> None:
        self._cache: dict[BlockHashWithGroupId, dict[int, KVCacheBlock]] = {}

    def get_one_block(
        self,
        key: BlockHashWithGroupId,
    ) -> KVCacheBlock | None:
        blocks = self._cache.get(key)
        return next(iter(blocks.values())) if blocks else None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        self._cache.setdefault(key, {})[block.block_id] = block

    def pop(
        self,
        key: BlockHashWithGroupId,
        block_id: int,
    ) -> KVCacheBlock | None:
        blocks = self._cache.get(key)
        if not blocks:
            return None
        block = blocks.pop(block_id, None)
        if not blocks:
            self._cache.pop(key, None)
        return block

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class BlockPool:
    """统一管理 block 分配、引用计数、LRU 驱逐和 Prefix Cache 索引。"""

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
    ) -> None:
        if num_gpu_blocks <= 1:
            raise ValueError("num_gpu_blocks 至少为 2")
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size

        self.blocks = [KVCacheBlock(i) for i in range(num_gpu_blocks)]
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        self.cached_block_hash_to_block = BlockHashToBlockMap()
        self.cached_block_hashes_by_block: dict[
            int, set[BlockHashWithGroupId]
        ] = {}

        # 与 vLLM 一致，保留一个永不缓存、永不计引用的占位 block。
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

    def get_cached_block(
        self,
        block_hash: BlockHash,
        kv_cache_group_ids: list[int],
    ) -> list[KVCacheBlock] | None:
        """只有所有指定 group 都命中，才返回这组共享前缀块。"""

        cached_blocks: list[KVCacheBlock] = []
        for group_id in kv_cache_group_ids:
            key = make_block_hash_with_group_id(block_hash, group_id)
            block = self.cached_block_hash_to_block.get_one_block(key)
            if block is None:
                return None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """登记模型已经真正计算完成的 full blocks。"""

        if not self.enable_caching or num_cached_blocks >= num_full_blocks:
            return
        request.update_block_hashes(self.hash_block_size)

        for block_index in range(num_cached_blocks, num_full_blocks):
            block = blocks[block_index]
            if block.is_null:
                continue
            prefix_tokens = (block_index + 1) * block_size
            hash_index = prefix_tokens // self.hash_block_size - 1
            block_hash = request.block_hashes[hash_index]
            key = make_block_hash_with_group_id(block_hash, kv_cache_group_id)
            self._insert_block_hash(key, block, prefix_tokens)

    def _insert_block_hash(
        self,
        key: BlockHashWithGroupId,
        block: KVCacheBlock,
        num_tokens: int,
    ) -> None:
        if block.block_hash == key:
            return
        if block.block_hash is None:
            block.set_block_hash(key, num_tokens)
        else:
            self.cached_block_hashes_by_block.setdefault(block.block_id, set()).add(
                key
            )
        self.cached_block_hash_to_block.insert(key, block)

    def _remove_cached_block_hashes(self, block: KVCacheBlock) -> None:
        keys: set[BlockHashWithGroupId] = set()
        if block.block_hash is not None:
            keys.add(block.block_hash)
        keys.update(self.cached_block_hashes_by_block.pop(block.block_id, set()))
        for key in keys:
            self.cached_block_hash_to_block.pop(key, block.block_id)
        block.reset_hash()

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(
                f"空闲 block 不足：需要 {num_blocks}，"
                f"只剩 {self.get_num_free_blocks()}"
            )
        blocks = self.free_block_queue.popleft_n(num_blocks)
        for block in blocks:
            if self.enable_caching:
                self._remove_cached_block_hashes(block)
            if block.ref_cnt != 0 or block.is_null:
                raise RuntimeError(f"block {block.block_id} 状态异常")
            block.ref_cnt = 1
        return blocks

    def touch(self, blocks: Iterable[KVCacheBlock]) -> None:
        """Prefix Cache 命中后，把驱逐候选重新变成被请求持有的块。"""

        for block in blocks:
            if block.is_null:
                continue
            if block.ref_cnt == 0:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1

    def free_blocks(self, blocks: Iterable[KVCacheBlock]) -> None:
        """按调用方给出的顺序释放；group manager 会传入反向块序列。"""

        for block in blocks:
            if block.is_null:
                continue
            if block.ref_cnt <= 0:
                raise RuntimeError(f"block {block.block_id} 引用计数异常")
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                # 尾部先 append，头部后 append：头部最终更靠近队尾、更晚驱逐。
                self.free_block_queue.append(block)

    def get_num_free_blocks(self) -> int:
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        usable_blocks = self.num_gpu_blocks - 1
        return 1.0 - self.get_num_free_blocks() / usable_blocks

    def reset_prefix_cache(self) -> bool:
        """仅在没有活动引用时清空 Prefix Cache。"""

        if any(block.ref_cnt > 0 for block in self.blocks if not block.is_null):
            return False
        self.cached_block_hash_to_block.clear()
        self.cached_block_hashes_by_block.clear()
        for block in self.blocks:
            block.reset_hash()
        return True
