"""
配置系统：EngineConfig + ParallelConfig + CLI 参数解析

对应 vLLM 的 vllm/config/parallel.py 里的 ParallelConfig 部分（简化版）。
"""

import argparse
from dataclasses import dataclass, field


@dataclass
class ParallelConfig:
    """并行配置 — 描述模型如何在多张 GPU 上切分

    对应 vLLM 的 ParallelConfig。

    单机多卡场景下的核心概念：
      - TP (tensor_parallel_size):   张量并行，把一层权重按列/行切到多张卡
      - PP (pipeline_parallel_size): 流水线并行，把模型按层切到多张卡
      - world_size = tp × pp         一个模型副本所需的 GPU 总数

    rank 布局（先 PP 后 TP 交织）：
      对于 rank 从 0 到 world_size-1：
        tp_rank = rank % tp_size   —— 同一 PP stage 内，第几张 TP 卡
        pp_rank = rank // tp_size  —— 第几个 PP stage

    例如 TP=2, PP=2, world_size=4：
        rank 0 → tp_rank=0, pp_rank=0
        rank 1 → tp_rank=1, pp_rank=0
        rank 2 → tp_rank=0, pp_rank=1
        rank 3 → tp_rank=1, pp_rank=1
    """

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    # 以下字段由运行期填充（Worker 进程启动时写入），不是 CLI 参数
    rank: int = 0        # 全局逻辑 rank（一个模型副本内的全局编号）
    local_rank: int = 0  # 本机逻辑 rank（单机多卡时 == rank）

    @property
    def world_size(self) -> int:
        """一个模型副本所需的 GPU 数 = tp × pp"""
        return self.tensor_parallel_size * self.pipeline_parallel_size

    @property
    def local_world_size(self) -> int:
        """本机节点上的 worker 数（单机多卡时 == world_size）"""
        return self.world_size


@dataclass
class EngineConfig:
    """引擎配置 — 从 CLI 参数构造"""

    host: str = "0.0.0.0"
    port: int = 8000
    model: str = "test-model"
    disable_log_stats: bool = False
    served_model_name: str | None = None
    max_model_len: int = 4096
    data_parallel_size: int = 1  # DP 并行度（引擎后端进程数）
    enable_log_requests: bool = False
    seed: int = 0  # 随机种子，保证推理链的确定性

    # ---- 模型加载配置（简化版 ModelConfig + LoadConfig）----
    load_format: str = "safetensors"
    dtype: str = "auto"

    # ---- KV cache 配置（简化版 CacheConfig）----
    block_size: int = 16        # 每个 KV block 容纳的 token 数
    # None 表示由 profiling 自动计算；显式指定仅用于调试/复现实验。
    num_gpu_blocks: int | None = None
    gpu_memory_utilization: float = 0.9
    kv_cache_memory_bytes: int | None = None
    kv_cache_dtype: str = "auto"
    enable_prefix_caching: bool = True  # 是否启用前缀缓存（prefix caching）

    # ---- 调度器配置（简化版 SchedulerConfig）----
    max_num_seqs: int = 32             # 同时处于 RUNNING 状态的请求数上限
    max_num_batched_tokens: int = 2048  # 单步调度（一个 batch）的 token 预算

    # ---- CUDA Graph（第一阶段只捕获纯 Decode 的整模型 forward）----
    enable_cuda_graph: bool = True
    cuda_graph_seq_len_bucket_size: int = 256
    cuda_graph_num_warmups: int = 1

    # 并行配置（TP/PP），预留给后续并行优化
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "EngineConfig":
        return cls(
            host=args.host,
            port=args.port,
            model=args.model,
            disable_log_stats=args.disable_log_stats,
            served_model_name=args.served_model_name,
            max_model_len=args.max_model_len,
            data_parallel_size=getattr(args, "dp_size", 1),
            enable_log_requests=getattr(args, "enable_log_requests", False),
            seed=getattr(args, "seed", 0),
            load_format=getattr(args, "load_format", "safetensors"),
            dtype=getattr(args, "dtype", "auto"),
            block_size=getattr(args, "block_size", 16),
            num_gpu_blocks=getattr(args, "num_gpu_blocks", None),
            gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.9),
            kv_cache_memory_bytes=getattr(args, "kv_cache_memory_bytes", None),
            kv_cache_dtype=getattr(args, "kv_cache_dtype", "auto"),
            enable_prefix_caching=not getattr(args, "disable_prefix_caching", False),
            max_num_seqs=getattr(args, "max_num_seqs", 32),
            max_num_batched_tokens=getattr(args, "max_num_batched_tokens", 2048),
            enable_cuda_graph=not getattr(args, "disable_cuda_graph", False),
            cuda_graph_seq_len_bucket_size=getattr(
                args, "cuda_graph_seq_len_bucket_size", 256
            ),
            cuda_graph_num_warmups=getattr(args, "cuda_graph_num_warmups", 1),
            parallel_config=ParallelConfig(
                tensor_parallel_size=getattr(args, "tp_size", 1),
                pipeline_parallel_size=getattr(args, "pp_size", 1),
            ),
        )


def make_arg_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数解析器"""
    parser = argparse.ArgumentParser(
        description="my_vllm — 极简在线推理服务"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="服务器监听地址"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="服务器监听端口"
    )
    parser.add_argument(
        "--model", type=str, default="test-model", help="模型名称或路径"
    )
    parser.add_argument(
        "--load-format",
        choices=("safetensors", "dummy"),
        default="safetensors",
        help="权重加载格式；dummy 只构造并随机初始化模型，用于结构调试",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="模型参数 dtype",
    )
    parser.add_argument(
        "--disable-log-stats",
        action="store_true",
        help="禁用定期统计日志",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="对外暴露的模型名称",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="模型最大上下文长度",
    )
    parser.add_argument(
        "--enable-log-requests",
        action="store_true",
        help="启用请求日志",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="数据并行度（DP，引擎后端进程数）",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="张量并行度（TP，一张模型切到几张卡）",
    )
    parser.add_argument(
        "--pp-size",
        type=int,
        default=1,
        help="流水线并行度（PP，模型按层切到几个 stage）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子（保证推理确定性）",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="KV cache 每个 block 容纳的 token 数",
    )
    parser.add_argument(
        "--num-gpu-blocks",
        type=int,
        default=None,
        help="手工覆盖 KV cache block 数；默认由 profiling 自动计算",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="单个 Worker 可使用的 GPU 总显存比例",
    )
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        default=None,
        help="手工指定 KV cache 字节预算；指定后仍跑 dummy forward，但跳过自动预算",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="KV cache dtype；auto 跟随模型 dtype",
    )
    parser.add_argument(
        "--disable-prefix-caching",
        action="store_true",
        help="禁用前缀缓存（prefix caching）",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=32,
        help="同时处于 RUNNING 状态的请求数上限",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=2048,
        help="单步调度（一个 batch）的 token 预算",
    )
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="禁用纯 Decode FULL CUDA Graph，所有 batch 均使用 eager forward",
    )
    parser.add_argument(
        "--cuda-graph-seq-len-bucket-size",
        type=int,
        default=256,
        help="纯 Decode 图的 max_seq_len 分桶步长",
    )
    parser.add_argument(
        "--cuda-graph-num-warmups",
        type=int,
        default=1,
        help="每种 FULL CUDA Graph 规格捕获前的旁路 stream warmup 次数",
    )
    return parser
