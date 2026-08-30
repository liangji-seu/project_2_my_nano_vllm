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
_PP_BROADCAST: ProcessGroup | None = None
_TP_SIZE = 1
_PP_SIZE = 1
_TP_RANK = 0
_PP_RANK = 0


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
    global _TP, _PP, _PP_BROADCAST, _TP_SIZE, _PP_SIZE, _TP_RANK, _PP_RANK

    assert dist.is_initialized(), "必须先调用 init_distributed_environment()"

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert world_size == tensor_parallel_size * pipeline_parallel_size, (
        f"world_size({world_size}) 必须等于 tp({tensor_parallel_size}) "
        f"× pp({pipeline_parallel_size})"
    )

    if tensor_parallel_size < 1 or pipeline_parallel_size < 1:
        raise ValueError("tensor_parallel_size 和 pipeline_parallel_size 必须 >= 1")

    tp_rank = rank % tensor_parallel_size
    pp_rank = rank // tensor_parallel_size
    _TP_SIZE = tensor_parallel_size
    _PP_SIZE = pipeline_parallel_size
    _TP_RANK = tp_rank
    _PP_RANK = pp_rank

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

    # new_group 是集合操作：每个 rank 都必须以完全相同的顺序创建全部子组。
    # 仅创建本 rank 所在的组会让不同 rank 调用次数不一致，最终造成 NCCL/Gloo 死锁。
    _TP = None
    for ranks in tp_groups:
        group = dist.new_group(ranks) if tensor_parallel_size > 1 else None
        if rank in ranks:
            _TP = group

    _PP = None
    _PP_BROADCAST = None
    for ranks in pp_groups:
        group = dist.new_group(ranks) if pipeline_parallel_size > 1 else None
        # 【ModelRunner V2·PP 旁路】相同成员再建一个 sibling communicator，
        # sampled-token broadcast 不和 activation P2P 共用 NCCL communicator。
        broadcast_group = (
            dist.new_group(ranks) if pipeline_parallel_size > 1 else None
        )
        if rank in ranks:
            _PP = group
            _PP_BROADCAST = broadcast_group

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


def get_pp_broadcast_group() -> ProcessGroup | None:
    return _PP_BROADCAST


def get_tensor_model_parallel_world_size() -> int:
    return _TP_SIZE


def get_tensor_model_parallel_rank() -> int:
    return _TP_RANK


def get_pipeline_model_parallel_world_size() -> int:
    return _PP_SIZE


def get_pipeline_model_parallel_rank() -> int:
    return _PP_RANK


def tensor_model_parallel_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """对 TP 组执行原地 SUM all-reduce，并返回同一张量。"""
    if _TP is not None:
        dist.all_reduce(tensor, group=_TP)
    return tensor


def tensor_model_parallel_all_gather(
    tensor: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """沿 ``dim`` 收集 TP rank 的局部张量，并按 rank 顺序拼接。

    ParallelLMHead 使用它把各 rank 的局部 vocabulary logits 恢复成完整
    vocabulary。学习版本优先保证采样语义完全正确；后续可以进一步实现
    distributed argmax，避免物化完整 logits。
    """

    if _TP is None:
        return tensor
    shards = [torch.empty_like(tensor) for _ in range(_TP_SIZE)]
    dist.all_gather(shards, tensor.contiguous(), group=_TP)
    return torch.cat(shards, dim=dim)


def is_pipeline_first_stage() -> bool:
    return _PP_RANK == 0


def is_pipeline_last_stage() -> bool:
    return _PP_RANK == _PP_SIZE - 1


def get_pipeline_model_parallel_prev_rank() -> int | None:
    """返回同一 TP lane 上的前一 PP stage 全局 rank。"""
    if _PP_RANK == 0:
        return None
    return (_PP_RANK - 1) * _TP_SIZE + _TP_RANK


def get_pipeline_model_parallel_next_rank() -> int | None:
    """返回同一 TP lane 上的后一 PP stage 全局 rank。"""
    if _PP_RANK == _PP_SIZE - 1:
        return None
    return (_PP_RANK + 1) * _TP_SIZE + _TP_RANK


def pipeline_model_parallel_send(tensor: torch.Tensor) -> None:
    """将 activation 发送给同一 TP lane 的下一 PP stage。"""
    destination = get_pipeline_model_parallel_next_rank()
    if destination is not None:
        dist.send(tensor.contiguous(), dst=destination)


def pipeline_model_parallel_isend(tensor: torch.Tensor):
    """异步发送 activation；调用方必须保活 tensor 并在复用前 wait。"""

    destination = get_pipeline_model_parallel_next_rank()
    if destination is None:
        return None
    if not tensor.is_contiguous():
        raise ValueError("PP isend tensor 必须 contiguous，调用方负责保活发送 buffer")
    return dist.isend(tensor, dst=destination)


def pipeline_model_parallel_recv(tensor: torch.Tensor) -> torch.Tensor:
    """从同一 TP lane 的上一 PP stage 原地接收 activation。"""
    source = get_pipeline_model_parallel_prev_rank()
    if source is not None:
        dist.recv(tensor, src=source)
    return tensor


def pipeline_model_parallel_irecv(tensor: torch.Tensor):
    """异步接收 activation；真正消费 tensor 前必须 wait。"""

    source = get_pipeline_model_parallel_prev_rank()
    if source is None:
        return None
    return dist.irecv(tensor, src=source)


def destroy_model_parallel() -> None:
    """销毁 TP/PP 通信组（worker 关闭时调用）"""
    global _TP, _PP, _PP_BROADCAST, _TP_SIZE, _PP_SIZE, _TP_RANK, _PP_RANK
    _TP = None
    _PP = None
    _PP_BROADCAST = None
    _TP_SIZE = _PP_SIZE = 1
    _TP_RANK = _PP_RANK = 0


def destroy_distributed_environment() -> None:
    """销毁 world 通信组（worker 关闭时调用）"""
    if dist.is_initialized():
        dist.destroy_process_group()
