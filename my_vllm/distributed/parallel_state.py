"""
简化版分布式初始化 — 单机多卡下的 torch.distributed + 模型并行分组

对应 vLLM 的 vllm/distributed/parallel_state.py

只保留单机多卡 + 预留 TP/PP 所需的最小功能：
  1. init_distributed_environment()  — 初始化 world 通信组（NCCL）
  2. init_model_parallel_group()     — 按 rank 切分 TP / PP 子通信组
  3. destroy_*()                     — 清理通信组

简化说明（相比 vLLM 源码）：
  - 省略 DP / EP / EPLB / PCP / DCP 等高级分组（当前不需要）
  - 省略自定义 all_reduce 算子、通信算子缓存、StatelessProcessGroup 等
"""

import logging

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

# 全局通信组（模块级单例）。vLLM 也这么做：进程内只有一份分组信息。
_TP: ProcessGroup | None = None
_PP: ProcessGroup | None = None


def init_distributed_environment(
    world_size: int,
    rank: int,
    distributed_init_method: str,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """初始化 torch.distributed 的 world 通信组

    对应 vLLM 的 init_distributed_environment()。

    这是 NCCL 网络的入口：所有 worker 进程各自调用一次，通过
    distributed_init_method（如 tcp://127.0.0.1:29500）互相握手，
    形成一个 world_size 个进程的通信组。

    Args:
        world_size:  一个模型副本的 worker 总数（= tp × pp）
        rank:        本进程在 world 组内的全局编号
        distributed_init_method: 握手地址（tcp://ip:port）
        local_rank:  本机逻辑卡号（单机时 == rank）
        backend:     通信后端，GPU 用 nccl，CPU 调试可换成 gloo
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed 不可用，无法初始化分布式环境")

    if not dist.is_initialized():
        # 每个 worker 进程独立调用，按 rank 加入同一个 world 组
        dist.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
        )

    logger.info(
        "world 通信组初始化完成 (world_size=%d, rank=%d, backend=%s)",
        world_size,
        rank,
        backend,
    )


def init_model_parallel_group(
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
) -> None:
    """切分 TP / PP 子通信组

    对应 vLLM 的 initialize_model_parallel()。

    rank 布局：tp_rank = rank % tp_size, pp_rank = rank // tp_size

    - TP 组：同一 PP stage 内、不同 TP 位置的 rank 归为一组
      例 TP=2, PP=2, world=4 → TP 组: [0,1] 和 [2,3]
    - PP 组：同一 TP 位置、不同 PP stage 的 rank 归为一组
      例 TP=2, PP=2, world=4 → PP 组: [0,2] 和 [1,3]
    """
    global _TP, _PP

    assert dist.is_initialized(), "必须先调用 init_distributed_environment()"

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert world_size == tensor_parallel_size * pipeline_parallel_size, (
        f"world_size({world_size}) 必须等于 tp({tensor_parallel_size}) "
        f"× pp({pipeline_parallel_size})"
    )

    tp_rank = rank % tensor_parallel_size
    pp_rank = rank // tensor_parallel_size

    # 构造所有 TP 组的 rank 列表：每 tp 个连续 rank 组成一个 PP stage
    tp_groups = [
        list(range(pp * tensor_parallel_size, (pp + 1) * tensor_parallel_size))
        for pp in range(pipeline_parallel_size)
    ]
    # 构造所有 PP 组的 rank 列表：相同 tp 位置、跨 stage
    pp_groups = [
        list(range(tp, world_size, tensor_parallel_size))
        for tp in range(tensor_parallel_size)
    ]

    # 创建本进程所在的通信组。
    # 注意：dist.new_group() 是集合操作，所有 rank 必须按相同顺序调用相同次数，
    # 且同一组内的成员必须传入相同的 rank 列表。
    if tensor_parallel_size > 1:
        _TP = dist.new_group(tp_groups[pp_rank])
    if pipeline_parallel_size > 1:
        _PP = dist.new_group(pp_groups[tp_rank])

    logger.info(
        "模型并行组初始化完成 (rank=%d, tp_rank=%d, pp_rank=%d)",
        rank,
        tp_rank,
        pp_rank,
    )


# ---- 访问器（供后续 TP/PP 通信逻辑使用）----


def get_tp_group() -> ProcessGroup | None:
    """返回本进程所在的 TP 通信组（TP=1 时为 None）"""
    return _TP


def get_pp_group() -> ProcessGroup | None:
    """返回本进程所在的 PP 通信组（PP=1 时为 None）"""
    return _PP


def get_tensor_model_parallel_world_size() -> int:
    return _TP.size() if _TP is not None else 1


def get_tensor_model_parallel_rank() -> int:
    return dist.get_rank(_TP) if _TP is not None else 0


def get_pipeline_model_parallel_world_size() -> int:
    return _PP.size() if _PP is not None else 1


def get_pipeline_model_parallel_rank() -> int:
    return dist.get_rank(_PP) if _PP is not None else 0


def destroy_model_parallel() -> None:
    """销毁 TP/PP 通信组（worker 关闭时调用）"""
    global _TP, _PP
    _TP = None
    _PP = None


def destroy_distributed_environment() -> None:
    """销毁 world 通信组（worker 关闭时调用）"""
    if dist.is_initialized():
        dist.destroy_process_group()
