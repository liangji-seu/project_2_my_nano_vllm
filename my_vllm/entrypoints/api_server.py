"""
API Server 核心流程：setup_server → build_engine → build_app → serve_http

完整启动链路（镜像 vLLM 的 api_server.py）:
  1. setup_server(args)        → 创建 socket，绑定端口
  2. AsyncLLM(vllm_config)     → 构造引擎前端实例
  3. build_app(args)           → 创建 FastAPI(lifespan=...)
  4. init_app_state(engine, state, args)  → 注入 engine 到 app.state
  5. serve_http(app, sock, ...)           → 启动 uvicorn
"""

import logging
import socket
from argparse import Namespace

from fastapi import FastAPI
from starlette.datastructures import State

from my_vllm.config import EngineConfig
from my_vllm.engine.async_llm import AsyncLLM
from my_vllm.engine.protocol import EngineClient
from my_vllm.entrypoints.server_utils import build_lifespan

logger = logging.getLogger(__name__)

# ---------- Socket ----------


def create_server_socket(sock_addr: tuple[str, int]) -> socket.socket:
    """
    创建并绑定 TCP socket

    先 bind 端口再初始化引擎，避免端口被抢占的竞态条件
    （和 vLLM 的做法一致，见 vllm#8204）
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(sock_addr)
    logger.info("Socket 已绑定: %s:%d", sock_addr[0], sock_addr[1])
    return sock


def setup_server(args: Namespace) -> tuple[str, socket.socket]:
    """
    初始化服务器基础设施

    返回:
        listen_address:  如 "http://0.0.0.0:8000"
        sock:            已 bind 的 socket
    """
    sock_addr = (args.host or "0.0.0.0", args.port)
    sock = create_server_socket(sock_addr)

    host_part = args.host or "0.0.0.0"
    listen_address = f"http://{host_part}:{args.port}"

    return listen_address, sock


# ---------- App 构造 ----------


def build_app(args: Namespace) -> FastAPI:
    """
    构造 FastAPI 应用

    对应 vLLM 的 build_app()，步骤：
    1. 创建 FastAPI 实例，挂载 lifespan
    2. 保存 args 到 app.state
    3. 注册所有路由
    """
    lifespan = build_lifespan()
    app = FastAPI(lifespan=lifespan)
    app.state.args = args

    # 注册 vLLM 风格的路由
    from my_vllm.routes.health import attach_router as attach_health_router
    from my_vllm.routes.chat import attach_router as attach_chat_router

    attach_health_router(app)
    attach_chat_router(app)

    return app


# ---------- State 注入 ----------


async def init_app_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
) -> None:
    """
    将引擎实例和配置注入到 app.state 中

    对应 vLLM 的 init_app_state()。
    后续接口函数通过 request.app.state.engine_client 获取引擎
    """
    vllm_config = engine_client.vllm_config

    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats
    state.vllm_config = vllm_config
    state.served_model_name = args.served_model_name or args.model

    logger.info(
        "App State 初始化完成: 模型=%s, log_stats=%s",
        state.served_model_name,
        state.log_stats,
    )


# ---------- 主流程 ----------


async def run_server(args: Namespace) -> None:
    """
    完整的 API Server 启动流程

    这是 my_vllm serve 命令的核心执行路径
    """
    # 1. 设置服务器基础设施
    listen_address, sock = setup_server(args)
    logger.info("服务器地址: %s", listen_address)

    # 2. 构造引擎配置
    vllm_config = EngineConfig.from_cli_args(args)

    # 3. 构造引擎前端（当前为占位实现）
    engine_client = AsyncLLM(vllm_config=vllm_config)

    try:
        # 4. 查询引擎支持的任务
        supported_tasks = await engine_client.get_supported_tasks()
        logger.info("支持的任务: %s", supported_tasks)

        # 5. 构造 FastAPI 应用
        app = build_app(args)

        # 6. 注入引擎到 app.state
        await init_app_state(engine_client, app.state, args)

        # 7. 启动 HTTP 服务
        from my_vllm.entrypoints.launcher import serve_http

        logger.info("正在启动 vLLM 风格服务器，监听 %s", listen_address)
        logger.info("按 Ctrl+C 停止服务")

        await serve_http(
            app,
            sock=sock,
            host=args.host,
            port=args.port,
            log_level="info",
        )
    finally:
        await engine_client.shutdown()
