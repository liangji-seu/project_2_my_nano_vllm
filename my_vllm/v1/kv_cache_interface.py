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

    @property
    def page_size_bytes(self) -> int:
        """一个 layer 的一个 block 同时存 K 和 V 所需的字节数。"""

        dtype_sizes = {
            "float16": 2,
            "torch.float16": 2,
            "bfloat16": 2,
            "torch.bfloat16": 2,
            "float32": 4,
            "torch.float32": 4,
        }
        try:
            element_size = dtype_sizes[self.dtype]
        except KeyError as exc:
            raise ValueError(f"无法计算 KV cache dtype={self.dtype!r} 的字节数") from exc
        return 2 * self.block_size * self.num_kv_heads * self.head_size * element_size


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


def generate_kv_cache_config(
    kv_cache_spec: dict[str, FullAttentionSpec],
    available_memory: int,
    num_blocks_override: int | None = None,
) -> KVCacheConfig:
    """把模型的逐层 KV 规格和显存预算转换成可执行的物理布局。"""

    if not kv_cache_spec:
        raise ValueError("模型没有返回任何 KV cache layer spec")
    specs = list(kv_cache_spec.values())
    first = specs[0]
    if any(spec != first for spec in specs[1:]):
        raise NotImplementedError("当前阶段只实现所有 attention layer 形状相同的单 group 模型")

    bytes_per_block = sum(spec.page_size_bytes for spec in specs)
    num_blocks = (
        num_blocks_override
        if num_blocks_override is not None
        else available_memory // bytes_per_block
    )
    if num_blocks <= 1:
        raise MemoryError(
            "KV cache 显存不足："
            f"available={available_memory} bytes, bytes_per_block={bytes_per_block}, "
            f"num_blocks={num_blocks}（至少需要 2，含 null block）"
        )
    required_memory = num_blocks * bytes_per_block
    if required_memory > available_memory:
        raise MemoryError(
            f"手工指定的 num_blocks={num_blocks} 需要 {required_memory} bytes，"
            f"超过 profiling 预算 {available_memory} bytes"
        )

    layer_names = list(kv_cache_spec)
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * kv_cache_spec[layer_name].page_size_bytes,
                shared_by=[layer_name],
            )
            for layer_name in layer_names
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=first)
        ],
    )
