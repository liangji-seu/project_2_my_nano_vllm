"""
引擎传输层 — 进程管理器 + 启动器

CoreEngineProcManager:  管理引擎后端进程的创建、启动、生命周期
launch_core_engines():  contextmanager，协调引擎进程的创建和清理

对应 vLLM 的 vllm/v1/engine/utils.py
"""

import contextlib
import logging
import multiprocessing
import weakref
from typing import Iterator

from my_vllm.config import EngineConfig

logger = logging.getLogger(__name__)


def shutdown_processes(processes: list[multiprocessing.Process]) -> None:
    """关闭所有引擎进程 — 先 SIGTERM，超时后 SIGKILL"""
    for proc in processes:
        if proc.is_alive():
            logger.info(
                "正在终止引擎进程: %s (pid=%d)", proc.name, proc.pid
            )
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                logger.warning("引擎进程未响应 SIGTERM，强制 kill: %s", proc.name)
                proc.kill()
                proc.join(timeout=2)


class CoreEngineProcManager:
    """引擎后端进程管理器

    职责:
      - 构造 multiprocessing.Process 对象（target=EngineCoreProc.run_engine_core）
      - 启动所有引擎进程
      - 通过 weakref.finalize 注册 GC 安全的清理回调

    对应 vLLM 的 CoreEngineProcManager
    """

    def __init__(
        self,
        engine_count: int,
        vllm_config: EngineConfig,
    ):
        # 使用 spawn 方式（macOS 默认），确保子进程干净启动
        ctx = multiprocessing.get_context("spawn")
        self.processes: list[multiprocessing.Process] = []

        from my_vllm.engine.core import EngineCoreProc

        common_kwargs: dict = {
            "vllm_config": vllm_config,
        }

        for i in range(engine_count):
            proc_name = f"EngineCore_{i}" if engine_count > 1 else "EngineCore"
            proc = ctx.Process(
                target=EngineCoreProc.run_engine_core,
                name=proc_name,
                kwargs=common_kwargs | {"engine_index": i},
            )
            self.processes.append(proc)
            logger.info("创建引擎进程对象: %s", proc.name)

        # 注册 GC 安全清理：当 CoreEngineProcManager 被 GC 时自动终止进程
        # 防止 __del__ 的循环引用问题
        self._finalizer = weakref.finalize(
            self, shutdown_processes, self.processes
        )

    def start_all(self) -> None:
        """启动所有引擎进程"""
        for proc in self.processes:
            proc.start()
            logger.info("引擎进程已启动: %s (pid=%d)", proc.name, proc.pid)


@contextlib.contextmanager
def launch_core_engines(
    vllm_config: EngineConfig,
    engine_count: int = 1,
) -> Iterator[CoreEngineProcManager]:
    """启动引擎后端进程的上下文管理器

    对应 vLLM 的 launch_core_engines()。

    使用 @contextmanager + yield 实现 try/finally 语义:
      before yield:  创建并启动引擎进程
      after yield:   服务运行中...
      finally:       清理引擎进程

    用法:
        with launch_core_engines(vllm_config, engine_count=1) as engine_manager:
            # 引擎进程已启动，这里执行服务逻辑
            serve()
        # 退出 with 块后自动终止引擎进程
    """
    logger.info("正在启动 %d 个引擎后端进程...", engine_count)

    manager = CoreEngineProcManager(engine_count, vllm_config)
    manager.start_all()

    # 给子进程一点时间完成初始化（后续 commit 改为 ZMQ 握手确认）
    import time

    time.sleep(1.0)
    logger.info("已启动 %d 个引擎后端进程", engine_count)

    try:
        yield manager
    finally:
        logger.info("正在关闭引擎后端进程...")
        shutdown_processes(manager.processes)
        logger.info("引擎后端进程已全部关闭")
