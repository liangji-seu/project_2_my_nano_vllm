"""
服务工具函数：lifespan（App 生命周期管理）
"""

import asyncio
import gc
import logging
from typing import TYPE_CHECKING

from fastapi import FastAPI

if TYPE_CHECKING:
    from my_vllm.engine.protocol import EngineClient

logger = logging.getLogger(__name__)

# 全局后台任务引用集 — 防止被 GC 回收
_running_tasks: set[asyncio.Task] = set()


def freeze_gc_heap() -> None:
    """
    冻结 GC 堆 — 启动期创建的静态对象不再被 GC 扫描
    减少服务运行时的 GC 暂停时间
    """
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    gc.freeze()
    logger.debug("GC 堆已冻结，启动期静态对象将不再被扫描")


def build_lifespan() -> callable:
    """
    构造 lifespan 函数

    返回一个符合 FastAPI lifespan 签名的异步上下文管理器函数。
    lifespan 中做的事情：
    - 启动阶段：开启定期统计日志任务
    - 关闭阶段：取消后台任务
    """

    @__import__("contextlib").asynccontextmanager
    async def lifespan(app: FastAPI):
        """App 生命周期管理"""
        try:
            if app.state.log_stats:
                engine_client: "EngineClient" = app.state.engine_client

                async def _force_log():
                    while True:
                        await asyncio.sleep(30)  # 每 30 秒输出一次统计
                        await engine_client.do_log_stats()

                task = asyncio.create_task(_force_log())
                _running_tasks.add(task)
                task.add_done_callback(_running_tasks.remove)
                logger.info("后台统计任务已启动（间隔 30s）")
            else:
                task = None

            freeze_gc_heap()

            try:
                yield  # ← 服务运行中
            finally:
                if task is not None:
                    task.cancel()
                    logger.info("后台统计任务已取消")
        finally:
            del app.state

    return lifespan
