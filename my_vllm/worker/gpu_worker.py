"""
GPU Worker 实现 — 单机多卡下真正的执行单元

对应 vLLM 的 vllm/v1/worker/gpu_worker.py

Worker 的三阶段初始化（每个 GPU 进程一份）：
  【阶段 1/3】init_device      — 在 worker 进程内直接执行：
                                 绑定 GPU + 拉起 NCCL + 构造 model_runner
  【阶段 2/3】load_model       — 在 worker 进程内直接执行：构造模型结构 + 加载权重
【阶段 3/3】initialize_from_config — 由 EngineCore 通过 collective_rpc 触发：
                                 分配 KV cache 显存

阶段 1/2 在 WorkerProc.__init__ 里被直接调用；阶段 3 是 EngineCore 主动 RPC。
"""

import gc
import logging
import random

import torch

from my_vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    init_model_parallel_group,
)
from my_vllm.worker.worker_base import WorkerBase

logger = logging.getLogger(__name__)


def set_random_seed(seed: int) -> None:
    """设置随机种子，保证整条推理链的确定性

    vLLM 里会用到多套随机数来源（Python random、numpy、torch、GPU 上的 CUDA 随机），
    全部设成同一个 seed，才能复现相同结果。
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Worker(WorkerBase):
    """GPU Worker — 每个 GPU 进程一个实例

    对应 vLLM 的 Worker（gpu_worker.py）
    """

    def init_device(self) -> None:
        """【Worker 初始化 · 阶段 1/3】Init Device

        步骤：
          1. 绑定本进程的 GPU 设备
          2. 拉起 NCCL 通信网络（world 组 + TP/PP 组）
          3. 设置随机种子
          4. 清理垃圾内存/显存（NCCL 会分配缓冲区，先清再测更准）
          5. 构造 model_runner
        """
        parallel_config = self.parallel_config

        # 1. 绑定 GPU。
        # 简化：单机多卡下，逻辑卡号 == 物理卡号（local_rank 即物理 GPU id）。
        # vLLM 里 local_rank 是「逻辑 id」，还要经过 assigned_physical_gpu_ids
        # 映射成「物理 id」；单机时两者一一对应，这里直接取 local_rank。
        visible_device_index = self.local_rank
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{visible_device_index}")
            # 真正把本进程绑定到这张卡（torch.device() 只是描述符，无副作用）
            torch.cuda.set_device(self.device)
        else:
            # 无 GPU：回退到 CPU 设备，便于本地无卡环境跑通整条 RPC 链路
            self.device = torch.device("cpu")
            logger.warning(
                "未检测到 CUDA，Worker rank=%d 使用 CPU 设备（本地无卡调试模式）",
                self.rank,
            )

        # 2. 拉起 NCCL：先 world 组，再切分 TP/PP 组
        self._init_worker_distributed_environment()

        # 3. 设置随机种子（保证推理确定性）
        set_random_seed(self.vllm_config.seed)

        # 4. 清理垃圾内存 / 显存
        gc.collect()
        torch.cuda.empty_cache()

        # 在模型加载前拍照。后续 profiling 会把「外部/NCCL 已占用 + 模型权重
        # + dummy forward 峰值」都从总预算中扣除。
        if self.device.type == "cuda":
            free_memory, total_memory = torch.cuda.mem_get_info(self.device)
        else:
            free_memory = total_memory = 0
        self.init_free_memory = free_memory
        self.total_memory = total_memory

        # 5. 构造 model_runner（本阶段已实现加载/profiling/KV，InputBatch 后续接入）
        from my_vllm.worker.gpu_model_runner import GPUModelRunner

        self.model_runner = GPUModelRunner(self.vllm_config, self.device)

        logger.info(
            "Worker rank=%d 完成【阶段 1/3】Init Device (device=%s)",
            self.rank,
            self.device,
        )

    def _init_worker_distributed_environment(self) -> None:
        """初始化分布式环境：world 组 + TP/PP 组

        对应 vLLM 的 init_worker_distributed_environment()
        """
        parallel_config = self.parallel_config
        world_size = parallel_config.world_size

        # world 组：一个模型副本的所有 worker 进程互相握手
        # 有 GPU 用 NCCL，无 GPU 用 gloo（CPU 后端，本地无卡可跑）
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        init_distributed_environment(
            world_size=world_size,
            rank=self.rank,
            distributed_init_method=self.distributed_init_method,
            local_rank=self.local_rank,
            backend=backend,
        )
        # TP/PP 子组：按 rank 切分（预留并行优化）
        init_model_parallel_group(
            tensor_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
        )

    def load_model(self) -> None:
        """【Worker 初始化 · 阶段 2/3】Load Model

        简化：直接委托给 model_runner。
        vLLM 里外面还包了「临时调 allocator 切片」「内存池」等加载环境准备。
        """
        self.model_runner.load_model()
        logger.info("Worker rank=%d 完成【阶段 2/3】Load Model", self.rank)

    def get_kv_cache_spec(self):
        """返回本 rank 模型逐层的 KV page 规格。"""
        return self.model_runner.get_kv_cache_spec()

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """profiling dummy forward，返回可分给 KV Cache 的字节数。"""
        if self.model_runner.is_mock_model:
            specs = self.model_runner.get_kv_cache_spec()
            blocks = self.vllm_config.num_gpu_blocks or 1024
            return blocks * sum(spec.page_size_bytes for spec in specs.values())
        if self.device.type != "cuda":
            raise RuntimeError("真实模型 profiling 需要 CUDA GPU")

        utilization = self.vllm_config.gpu_memory_utilization
        if not 0 < utilization <= 1:
            raise ValueError("gpu_memory_utilization 必须在 (0, 1] 范围内")

        gc.collect()
        torch.cuda.empty_cache()
        free_after_load, _ = torch.cuda.mem_get_info(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        allocated_before_profile = torch.cuda.memory_allocated(self.device)
        self.model_runner.profile_run()
        peak_allocated = torch.cuda.max_memory_allocated(self.device)
        activation_peak = max(0, peak_allocated - allocated_before_profile)

        requested_memory = int(self.total_memory * utilization)
        memory_used_before_profile = self.total_memory - free_after_load
        non_kv_cache_memory = memory_used_before_profile + activation_peak
        available = requested_memory - non_kv_cache_memory
        manual = self.vllm_config.kv_cache_memory_bytes
        if manual is not None:
            available = manual
        if available <= 0:
            raise MemoryError(
                "profiling 后没有可用 KV cache 显存："
                f"total={self.total_memory}, utilization={utilization}, "
                f"used_before_profile={memory_used_before_profile}, "
                f"activation_peak={activation_peak}, available={available}"
            )
        logger.info(
            "显存 profiling：total=%.2f GiB, requested=%.2f GiB, "
            "weights=%.2f GiB, activation_peak=%.2f GiB, available_kv=%.2f GiB",
            self.total_memory / 1024**3,
            requested_memory / 1024**3,
            self.model_runner.model_memory_usage / 1024**3,
            activation_peak / 1024**3,
            available / 1024**3,
        )
        gc.collect()
        torch.cuda.empty_cache()
        return available

    def initialize_from_config(self, kv_cache_configs):
        """【Worker 初始化 · 阶段 3/3】Initialize KV Cache

        由 EngineCore 通过 collective_rpc 触发，为每个 worker 分配 KV cache 显存。
        按 EngineCore 计算出的物理布局真实分配、reshape 并绑定到 attention layer。
        """
        kv_cache_config = kv_cache_configs[self.rank]
        self.num_gpu_blocks = kv_cache_config.num_blocks
        self.model_runner.initialize_kv_cache(kv_cache_config)
        # 【CUDA Graph Capture】KV Cache 与 Dispatcher 合法 key 库就绪后，
        # 在 Worker 启动阶段主动 dummy_run 捕获，真实请求期间只 replay/eager。
        self.model_runner.capture_model()
        logger.info(
            "Worker rank=%d 完成【阶段 3/3】Initialize KV Cache (blocks=%d)",
            self.rank,
            kv_cache_config.num_blocks,
        )
        # 回传 ack，供 EngineCore 确认本 worker 已完成
        return self.rank

    def execute_model(self, scheduler_output):
        """执行一次模型前向（委托 model_runner）"""
        return self.model_runner.execute_model(scheduler_output)

    def check_health(self) -> str:
        """健康检查（用于验证 RPC 通路是否打通）"""
        return f"ok (rank={self.rank})"

    def get_model(self):
        return self.model_runner.get_model()

    def shutdown(self) -> None:
        """清理分布式环境（worker 退出时调用）"""
        destroy_model_parallel()
        destroy_distributed_environment()
