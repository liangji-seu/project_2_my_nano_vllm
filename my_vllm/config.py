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
            data_parallel_size=getattr(args, "data_parallel_size", 1),
            enable_log_requests=getattr(args, "enable_log_requests", False),
            seed=getattr(args, "seed", 0),
            parallel_config=ParallelConfig(
                tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
                pipeline_parallel_size=getattr(args, "pipeline_parallel_size", 1),
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
        "--data-parallel-size",
        type=int,
        default=1,
        help="数据并行度（引擎后端进程数）",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="张量并行度（TP，一张模型切到几张卡）",
    )
    parser.add_argument(
        "--pipeline-parallel-size",
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
    return parser
