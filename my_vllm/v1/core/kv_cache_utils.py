"""KV cache 的 block 元数据、链式哈希键和空闲链表。"""

from dataclasses import dataclass
from typing import NewType

BlockHash = NewType("BlockHash", bytes)
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)


def make_block_hash_with_group_id(
    block_hash: BlockHash,
    group_id: int,
) -> BlockHashWithGroupId:
    """把内容链式 hash 与 group id 合成 Prefix Cache 的完整键。"""

    return BlockHashWithGroupId(
        block_hash + group_id.to_bytes(4, byteorder="big", signed=False)
    )


def get_block_hash(key: BlockHashWithGroupId) -> BlockHash:
    return BlockHash(key[:-4])


def get_group_id(key: BlockHashWithGroupId) -> int:
    return int.from_bytes(key[-4:], byteorder="big", signed=False)


@dataclass(slots=True)
class KVCacheBlock:
    """一个抽象 block 的 CPU 侧账本。

    ``block_id`` 会原样作为 Worker KV tensor 第一维的下标；block 自身永远
    不携带 head 数、head size 等形状信息，形状属于 KVCacheSpec。
    """

    block_id: int
    ref_cnt: int = 0
    _block_hash: BlockHashWithGroupId | None = None
    _block_hash_num_tokens: int | None = None
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None
    is_null: bool = False

    @property
    def block_hash(self) -> BlockHashWithGroupId | None:
        return self._block_hash

    @property
    def block_hash_num_tokens(self) -> int | None:
        return self._block_hash_num_tokens

    def set_block_hash(
        self,
        block_hash: BlockHashWithGroupId,
        num_tokens: int,
    ) -> None:
        if self._block_hash is not None:
            raise RuntimeError(f"block {self.block_id} 已经注册过 hash")
        self._block_hash = block_hash
        self._block_hash_num_tokens = num_tokens

    def reset_hash(self) -> None:
        self._block_hash = None
        self._block_hash_num_tokens = None


class FreeKVCacheBlockQueue:
    """以双向链表维护空闲块和 ref_cnt=0 的 Prefix Cache 驱逐候选。

    队头是最久未使用的 block。与 ``deque`` 版本相比，中间 block 被 Prefix
    Cache 命中时可以 O(1) 摘除。
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = 0
        self.fake_free_list_head = KVCacheBlock(-1)
        self.fake_free_list_tail = KVCacheBlock(-1)
        self.fake_free_list_head.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = self.fake_free_list_head
        for block in blocks:
            self.append(block)

    def popleft(self) -> KVCacheBlock:
        first = self.fake_free_list_head.next_free_block
        if first is None or first is self.fake_free_list_tail:
            raise ValueError("没有可用的 KV cache block")
        self.remove(first)
        return first

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        if n < 0 or n > self.num_free_blocks:
            raise ValueError(f"无法从 {self.num_free_blocks} 个空闲块中取 {n} 个")
        return [self.popleft() for _ in range(n)]

    def remove(self, block: KVCacheBlock) -> None:
        prev_block = block.prev_free_block
        next_block = block.next_free_block
        if prev_block is None or next_block is None:
            raise RuntimeError(f"block {block.block_id} 不在空闲链表中")
        prev_block.next_free_block = next_block
        next_block.prev_free_block = prev_block
        block.prev_free_block = None
        block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        if block.prev_free_block is not None or block.next_free_block is not None:
            raise RuntimeError(f"block {block.block_id} 已经在空闲链表中")
        previous = self.fake_free_list_tail.prev_free_block
        assert previous is not None
        previous.next_free_block = block
        block.prev_free_block = previous
        block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = block
        self.num_free_blocks += 1

    def prepend(self, block: KVCacheBlock) -> None:
        if block.prev_free_block is not None or block.next_free_block is not None:
            raise RuntimeError(f"block {block.block_id} 已经在空闲链表中")
        following = self.fake_free_list_head.next_free_block
        assert following is not None
        self.fake_free_list_head.next_free_block = block
        block.prev_free_block = self.fake_free_list_head
        block.next_free_block = following
        following.prev_free_block = block
        self.num_free_blocks += 1
