"""KV cache 配置协议。

这一层只描述 Worker 侧显存的规格和布局，不负责 block 的分配。
Scheduler 使用的 CPU 账本由 ``KVCacheManager`` 及其下层对象维护。
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KVCacheSpec:
    """一个 KV cache group 中单个物理 block 的形状规格。"""

    block_size: int

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size 必须大于 0")


@dataclass(frozen=True)
class FullAttentionSpec(KVCacheSpec):
    """全注意力 KV cache 的形状描述。"""

    num_kv_heads: int = 0
    head_size: int = 0
    dtype: str = "unknown"


@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    """Mamba 状态缓存规格占位，用于体现不同 group 可有不同形状。"""

    state_size: int = 0
    dtype: str = "unknown"


@dataclass(frozen=True)
class KVCacheTensor:
    """Worker 侧一片 KV cache 显存区域的布局描述。"""

    size: int
    shared_by: list[str] = field(default_factory=list)
    offset: int = 0
    block_stride: int = 0


@dataclass(frozen=True)
class KVCacheGroupSpec:
    """共享同一张 block table 的一组模型层及其 KV 形状。"""

    layer_names: list[str]
    kv_cache_spec: KVCacheSpec
    is_eagle_group: bool = False


@dataclass(frozen=True)
class KVCacheConfig:
    """Worker profiling 后交给 EngineCore 的完整 KV cache 配置。"""

    num_blocks: int
    kv_cache_tensors: list[KVCacheTensor]
    kv_cache_groups: list[KVCacheGroupSpec]

    def __post_init__(self) -> None:
        if self.num_blocks <= 1:
            raise ValueError("num_blocks 至少为 2（其中一个保留为 null block）")
        if not self.kv_cache_groups:
            raise ValueError("至少需要一个 KV cache group")

    @property
    def needs_kv_cache_zeroing(self) -> bool:
        """Mamba 状态块复用前需要清零；当前仅保留这一配置语义。"""

        return any(
            isinstance(group.kv_cache_spec, MambaSpec)
            for group in self.kv_cache_groups
        )


def make_default_kv_cache_config(
    num_blocks: int,
    block_size: int,
) -> KVCacheConfig:
    """构造普通 decoder-only Transformer 的单 group 配置。"""

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[KVCacheTensor(size=0, shared_by=[])],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[],
                kv_cache_spec=FullAttentionSpec(block_size=block_size),
            )
        ],
    )
