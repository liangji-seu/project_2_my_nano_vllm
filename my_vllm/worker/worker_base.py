"""
Worker 抽象层：WorkerBase 接口 + WorkerWrapperBase 适配转发

对应 vLLM 的 vllm/v1/worker/worker_base.py

两个类的分工：
  - WorkerBase：         定义「一个 GPU worker 该有哪些方法」的抽象接口
  - WorkerWrapperBase：  适配转发层，懒加载真正的 Worker，并利用 __getattr__
                        把未定义的方法/属性转发给内部 self.worker
"""

import torch
import torch.nn as nn


class WorkerBase:
    """Worker 抽象接口 — 定义 GPU worker 应有的方法

    对应 vLLM 的 WorkerBase。真实实现见 gpu_worker.py 的 Worker。
    """

    def __init__(
        self,
        vllm_config,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        """
        Args:
            vllm_config:            引擎配置（EngineConfig）
            local_rank:             本机逻辑卡号（单机时 == rank）
            rank:                   全局逻辑 rank（一个模型副本内）
            distributed_init_method: NCCL 握手地址（tcp://ip:port）
            is_driver_worker:        是否 TP 组的 driver worker（预留，TP>1 时负责加载权重）
        """
        self.vllm_config = vllm_config
        self.parallel_config = vllm_config.parallel_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        # 设备与模型状态（由子类在 init_device / load_model 中填充）
        self.device: torch.device | None = None
        self.model_runner: nn.Module | None = None

    # ---- 抽象接口（子类必须实现）----

    def init_device(self) -> None:
        """【阶段 1/3】初始化设备 + 分布式环境 + 构造 model_runner"""
        raise NotImplementedError

    def load_model(self) -> None:
        """【阶段 2/3】构造模型结构 + 加载权重"""
        raise NotImplementedError

    def execute_model(self, scheduler_output):
        """执行一次模型前向（推理主路径）"""
        raise NotImplementedError

    def sample_tokens(self):
        """消费最近一次 execute_model 的 GPU 状态并完成采样。"""
        raise NotImplementedError

    def initialize_from_config(self, kv_cache_config):
        """【阶段 3/3】初始化 KV cache（由 EngineCore 通过 collective_rpc 触发）"""
        raise NotImplementedError

    def check_health(self) -> None:
        """健康检查（默认无操作，子类可覆盖）"""
        return

    def get_model(self) -> nn.Module:
        raise NotImplementedError

    def shutdown(self) -> None:
        """清理资源（默认无操作）"""
        return


class WorkerWrapperBase:
    """Worker 的适配转发层

    对应 vLLM 的 WorkerWrapperBase。

    为什么需要这一层：
      WorkerProc 通过它懒加载真正的 Worker（GpuWorker 等），并用 __getattr__
      把未定义的方法调用转发给内部 self.worker。这样 WorkerProc 可以统一用
      wrapper.init_device() / wrapper.load_model() 等接口，而不关心底层具体
      是哪种 Worker。

    关键点：wrapper 自己定义的方法（init_worker / init_device / shutdown）优先级
    高于 __getattr__ 转发；其余方法（load_model / execute_model / initialize_from_config
    等）直接透传给 self.worker。
    """

    def __init__(self, rpc_rank: int = 0, global_rank: int | None = None) -> None:
        self.rpc_rank = rpc_rank  # 在本 executor 内的编号（单机时 == rank）
        self.global_rank = rpc_rank if global_rank is None else global_rank

        # 懒加载：init_worker 调用后才真正创建
        self.worker: WorkerBase | None = None

    def init_worker(self, all_kwargs: list[dict]) -> None:
        """构造真正的 Worker 子类实例

        对应 vLLM 的 init_worker()。

        Args:
            all_kwargs: 长度为 world_size 的列表，每个元素是给对应 rank 的
                        worker 的构造参数。本进程只取自己 rpc_rank 那份。
                        （vLLM 传 all_kwargs 是为了支持多 executor 共享一个
                        worker 的场景；单机下只有自己那份有值）
        """
        kwargs = all_kwargs[self.rpc_rank]
        vllm_config = kwargs.get("vllm_config")
        assert vllm_config is not None, "缺少 vllm_config 参数"

        # 简化：直接固定使用 GPU Worker。
        # vLLM 通过配置里的 worker_cls 字符串解析出真正的 Worker 子类，
        # 这里学习项目只有一个 GPU 实现，直接 import。
        from my_vllm.worker.gpu_worker import Worker

        self.worker = Worker(**kwargs)

    def init_device(self) -> None:
        """【阶段 1/3】转发给内部 worker 初始化设备"""
        return self.worker.init_device()

    def shutdown(self) -> None:
        """转发关闭"""
        if self.worker is not None:
            self.worker.shutdown()

    def __getattr__(self, attr: str):
        """未定义的属性/方法 → 转发给内部 self.worker

        注意：只有常规属性查找失败时才会调用 __getattr__，所以 wrapper
        自己定义的方法（init_worker / init_device / shutdown）不会被转发。
        """
        return getattr(self.worker, attr)
