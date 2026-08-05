"""
HTTP 启动器：用 uvicorn 在已绑定的 socket 上启动 FastAPI
"""

import asyncio
import logging
import signal
import socket
from functools import partial

import uvicorn
from fastapi import FastAPI

from my_vllm.engine.protocol import EngineClient

logger = logging.getLogger(__name__)


async def serve_http(
    app: FastAPI,
    sock: socket.socket | None = None,
    **uvicorn_kwargs,
):
    """
    在已绑定的 socket 上启动 uvicorn HTTP 服务器

    与 vLLM 的 serve_http 一致：
    - 打印所有已注册路由
    - 创建 uvicorn.Server，绑定到已有 socket
    - 启动 watchdog 和 graceful shutdown 机制

    参数:
        app: FastAPI 应用实例
        sock: 已 bind 的 socket 对象
        **uvicorn_kwargs: 传递给 uvicorn.Config 的额外参数
    """
    # 打印已注册路由
    logger.info("已注册路由:")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if path is not None:
            if methods:
                logger.info("  %s  %s", ", ".join(methods), path)
            else:
                logger.info("  %s", path)

    # 构造 uvicorn 配置
    config = uvicorn.Config(app, **uvicorn_kwargs)
    config.load()
    server = uvicorn.Server(config)
    app.state.server = server

    loop = asyncio.get_running_loop()

    # 启动 watchdog 任务 — 监控引擎是否崩溃
    watchdog_task = loop.create_task(_watchdog_loop(server, app.state.engine_client))

    # 启动 uvicorn 服务
    sockets = [sock] if sock else None
    server_task = loop.create_task(server.serve(sockets=sockets))

    # 优雅关闭处理
    shutdown_event = asyncio.Event()

    def signal_handler() -> None:
        shutdown_event.set()

    async def dummy_shutdown() -> None:
        pass

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    async def handle_shutdown() -> None:
        await shutdown_event.wait()

        engine_client: EngineClient = app.state.engine_client
        logger.info("收到关闭信号，正在清理...")

        await loop.run_in_executor(
            None, partial(engine_client.shutdown, timeout=30)
        )

        server.should_exit = True
        server_task.cancel()
        watchdog_task.cancel()

    shutdown_task = loop.create_task(handle_shutdown())

    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        logger.info("FastAPI HTTP 服务器已关闭")
        return server.shutdown()
    finally:
        shutdown_task.cancel()
        watchdog_task.cancel()


async def _watchdog_loop(server: uvicorn.Server, engine: EngineClient):
    """
    后台 Watchdog 任务：每 5 秒检查引擎是否崩溃
    如果引擎异常退出且未开启 keep-alive，则关闭 HTTP 服务
    """
    WATCHDOG_INTERVAL = 5.0
    while True:
        await asyncio.sleep(WATCHDOG_INTERVAL)

        engine_errored = engine.errored and not getattr(
            engine, "is_running", True
        )
        if engine_errored:
            logger.error("引擎异常退出，关闭 HTTP 服务")
            server.should_exit = True
