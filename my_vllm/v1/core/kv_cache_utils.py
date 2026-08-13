"""
KV cache 物理块基础设施 — KVCacheBlock + FreeKVCacheBlockQueue + BlockPool

对应 vLLM 的 vllm/v1/core/kv_cache_utils.py + block_pool.py（大幅简化版）。

核心概念：
  - 把 GPU 上的一大块 KV cache 显存切成等大的「block」，每个 block 装 block_size 个 token 的 KV。
  - BlockPool 统一管理这些 block 的分配 / 释放 / 引用计数，并维护「前缀缓存」：
    用 block 内容的 hash 做 key，命中即可复用已算好的 KV，避免重复计算。

简化说明（相比 vLLM）：
  - FreeKVCacheBlockQueue 用 collections.deque 实现，remove 是 O(n)；
    vLLM 用双向链表做到 O(1)，学习项目先不追求这个常数。
  - BlockHash 直接用 token id 元组 tuple[int, ...]，省掉 block_hasher 机制。
  - 省略 null_block（滑动窗口 / padding 用）、KV 事件、metrics 收集器。
"""

import logging
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# block 内容 hash：直接复用 token id 元组（tuple 本身可哈希，作为前缀缓存 key）
BlockHash = tuple[int, ...]


@dataclass(slots=True)
class KVCacheBlock:
    """KV cache 块元数据

    对应 vLLM 的 KVCacheBlock。
    """

    block_id: int
    ref_cnt: int = 0                       # 引用计数（多个请求共享同一前缀 block 时 +1）
    _block_hash: BlockHash | None = None   # 内容 hash，命中前缀缓存时非空
    is_null: bool = False                  # 占位块（本简化版未使用，保留字段）

    @property
    def block_hash(self) -> BlockHash | None:
        return self._block_hash

    def set_block_hash(self, block_hash: BlockHash) -> None:
        """写入内容 hash（只在 block 首次写满时调用）"""
        assert self._block_hash is None, "block 已存在 hash，不应重复设置"
        self._block_hash = block_hash

    def reset_hash(self) -> None:
        """block 被驱逐出前缀缓存时清空 hash"""
        self._block_hash = None

    def __repr__(self) -> str:
        return f"KVCacheBlock(block_id={self.block_id}, ref_cnt={self.ref_cnt})"


class FreeKVCacheBlockQueue:
    """空闲 block 队列

    对应 vLLM 的 FreeKVCacheBlockQueue。约定「最久未使用（LRU）的 block 在队头」，
    这样分配时 popleft 就能优先驱逐最久没用的 block。

    简化：用 deque 实现。remove(block) 是 O(n)，vLLM 用双向链表做到 O(1)。
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self._queue: deque[KVCacheBlock] = deque(blocks)
        self.num_free_blocks = len(blocks)

    def popleft(self) -> KVCacheBlock:
        block = self._queue.popleft()
        self.num_free_blocks -= 1
        return block

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        blocks = [self.popleft() for _ in range(n)]
        return blocks

    def remove(self, block: KVCacheBlock) -> None:
        """从队列中移除指定 block（O(n)）"""
        self._queue.remove(block)
        self.num_free_blocks -= 1

    def prepend(self, block: KVCacheBlock) -> None:
        """放到队头（成为 LRU 首选驱逐对象）"""
        self._queue.appendleft(block)
        self.num_free_blocks += 1

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        return list(self._queue)


class BlockPool:
    """物理 block 池：分配 / 释放 / 引用计数 + 前缀缓存索引

    对应 vLLM 的 BlockPool（简化版）。

    Args:
        num_gpu_blocks: block 总数（由显存估算得出）。
        enable_caching: 是否启用前缀缓存。
    """

    def __init__(self, num_gpu_blocks: int, enable_caching: bool):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching

        # 所有 block 元数据，block_id 即下标
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(block_id=i) for i in range(num_gpu_blocks)
        ]
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # 前缀缓存：block 内容 hash -> 已缓存 block（ref_cnt 可能为 0，属驱逐候选）
        self.cached_block_hash_to_block: dict[BlockHash, KVCacheBlock] = {}

    # ---- 前缀缓存 ----

    def get_cached_block(self, block_hash: BlockHash) -> KVCacheBlock | None:
        """按 hash 查前缀缓存，命中返回 block，否则 None"""
        if not self.enable_caching:
            return None
        return self.cached_block_hash_to_block.get(block_hash)

    def cache_block(self, block_hash: BlockHash, block: KVCacheBlock) -> None:
        """把 block 注册进前缀缓存"""
        if not self.enable_caching:
            return
        self.cached_block_hash_to_block[block_hash] = block

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> None:
        """block 被重新分配前，若它仍在前缀缓存里，则将其驱逐"""
        if block.block_hash is None:
            return
        self.cached_block_hash_to_block.pop(block.block_hash, None)
        block.reset_hash()

    # ---- 分配 / 释放 ----

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """从空闲队列取 num_blocks 个新 block（不检查前缀缓存，直接分配）

        注意：被取出的 block 若之前是「驱逐候选」（ref_cnt==0 但仍带 hash），
        需要先把它从前缀缓存驱逐掉，再交出去复用。
        """
        if num_blocks > self.free_block_queue.num_free_blocks:
            raise ValueError(f"空闲 block 不足：需要 {num_blocks}，只剩 "
                             f"{self.free_block_queue.num_free_blocks}")
        blocks = self.free_block_queue.popleft_n(num_blocks)
        for block in blocks:
            self._maybe_evict_cached_block(block)
            assert block.ref_cnt == 0
            block.ref_cnt += 1
        return blocks

    def touch(self, blocks: list[KVCacheBlock]) -> None:
        """前缀命中时给 block 引用计数 +1，若其原本在空闲队列则移出

        对应 vLLM 的 BlockPool.touch()：ref_cnt==0 的 block 还挂在空闲队列里
        （驱逐候选），被命中后要移出空闲队列（ref_cnt 变成 1，不再可被驱逐）。
        """
        for block in blocks:
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1

    def free_blocks(self, ordered_blocks: list[KVCacheBlock]) -> None:
        """释放一组 block，引用计数 -1；归零则放回空闲队列（队头 = 最先驱逐）

        参数需按「驱逐优先级」排序（首个最先驱逐）。调用方会按逆序传入，
        使序列尾部的 block 先被回收（尾块更晚使用，LRU 更「新」）。
        """
        for block in reversed(ordered_blocks):
            assert block.ref_cnt > 0, f"block {block.block_id} 引用计数异常"
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                # 归零仍保留 hash（懒驱逐）：它成为驱逐候选，分配时再真正驱逐
                self.free_block_queue.prepend(block)

    # ---- 查询 ----

    def get_num_free_blocks(self) -> int:
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """KV cache 使用率（0.0 ~ 1.0）"""
        return 1.0 - self.free_block_queue.num_free_blocks / self.num_gpu_blocks
