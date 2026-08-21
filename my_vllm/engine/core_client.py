"""
引擎传输层 — ZMQ sockets + 进程管理器

分层:
  get_engine_zmq_addresses():  生成 ZMQ IPC 地址（引擎前后端通信）
  CoreEngineProcManager:       管理引擎后端进程的生命周期
  launch_core_engines():       contextmanager, 协调引擎进程创建/清理
  MPClient:                    前端传输层 — ZMQ ROUTER/PULL + 进程管理

对应 vLLM 的 vllm/v1/engine/core_client.py + utils.py
"""

import asyncio
import contextlib
import json
import logging
import multiprocessing
import tempfile
import uuid
import weakref
from typing import Any, Iterator

import zmq
import zmq.asyncio  # noqa: F401 — 必须显式导入才能使用 zmq.asyncio.Context

from my_vllm.config import EngineConfig

logger = logging.getLogger(__name__)


# ==================================================================
# ZMQ 地址生成
# ==================================================================


def get_engine_zmq_addresses() -> dict[str, str]:
    """生成引擎前后端通信的 ZMQ IPC 地址

    使用 IPC (Unix domain socket) 协议:
      - 无端口冲突
      - 低延迟（不走 TCP 栈）
      - 单机专用

    后续迁移到多机 4090 服务器时改为 TCP

    Returns:
        {"input": "ipc:///tmp/my_vllm_xxx_input.ipc",
         "output": "ipc:///tmp/my_vllm_xxx_output.ipc"}
    """
    tmp_dir = tempfile.gettempdir()
    uid = uuid.uuid4().hex[:8]
    return {
        "input": f"ipc://{tmp_dir}/my_vllm_{uid}_input.ipc",
        "output": f"ipc://{tmp_dir}/my_vllm_{uid}_output.ipc",
    }


# ==================================================================
# 进程清理
# ==================================================================


def shutdown_processes(processes: list[multiprocessing.Process]) -> None:
    """关闭所有引擎进程 — 先 SIGTERM, 超时后 SIGKILL"""
    for proc in processes:
        if proc.is_alive():
            logger.info(
                "正在终止引擎进程: %s (pid=%d)", proc.name, proc.pid
            )
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                logger.warning(
                    "引擎进程未响应 SIGTERM, 强制 kill: %s", proc.name
                )
                proc.kill()
                proc.join(timeout=2)


# ==================================================================
# 进程管理器
# ==================================================================


class CoreEngineProcManager:
    """引擎后端进程管理器

    职责:
      - 构造 multiprocessing.Process 对象
      - 启动所有引擎进程
      - 通过 weakref.finalize 注册 GC 安全的清理回调

    对应 vLLM 的 CoreEngineProcManager
    """

    def __init__(
        self,
        engine_count: int,
        vllm_config: EngineConfig,
        input_address: str,
        output_address: str,
    ):
        # 使用 spawn 方式（macOS 默认）, 确保子进程干净启动
        ctx = multiprocessing.get_context("spawn")
        self.processes: list[multiprocessing.Process] = []

        from my_vllm.engine.core import EngineCoreProc

        common_kwargs: dict = {
            "vllm_config": vllm_config,
            "input_address": input_address,
            "output_address": output_address,
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

        # 注册 GC 安全清理回调
        # 防止 __del__ 的循环引用导致 GC 无法回收
        self._finalizer = weakref.finalize(
            self, shutdown_processes, self.processes
        )

    def start_all(self) -> None:
        """启动所有引擎进程"""
        for proc in self.processes:
            proc.start()
            logger.info("引擎进程已启动: %s (pid=%d)", proc.name, proc.pid)


# ==================================================================
# 引擎进程启动器 (contextmanager)
# ==================================================================


@contextlib.contextmanager
def launch_core_engines(
    vllm_config: EngineConfig,
    input_address: str,
    output_address: str,
    engine_count: int = 1,
) -> Iterator[CoreEngineProcManager]:
    """启动引擎后端进程的上下文管理器

    对应 vLLM 的 launch_core_engines()

    使用 @contextmanager + yield 实现 try/finally 语义:
      before yield:  创建并启动引擎进程
      after yield:   服务运行中...
      finally:       清理引擎进程
    """
    logger.info("正在启动 %d 个引擎后端进程...", engine_count)
    logger.info("ZMQ 地址: input=%s, output=%s", input_address, output_address)

    manager = CoreEngineProcManager(
        engine_count, vllm_config, input_address, output_address
    )
    manager.start_all()

    try:
        yield manager
    finally:
        logger.info("正在关闭引擎后端进程...")
        shutdown_processes(manager.processes)
        logger.info("引擎后端进程已全部关闭")


# ==================================================================
# 传输层 — MPClient
# ==================================================================


class MPClient:
    """前端传输层 — ZMQ sockets + 引擎进程管理

    对应 vLLM 的 MPClient (core_client.py)

    ZMQ 通信架构:
      ROUTER (bind)  → DEALER (connect)  引擎接收请求
      PULL   (bind)  ← PUSH   (connect)  引擎发送输出

    生命周期:
      1. __init__()    创建 ZMQ sockets + 启动引擎进程（同步）
      2. start()       握手等待引擎 READY + 启动输出处理任务（异步）
      3. generate()    发送请求 → 等待结果
      4. shutdown()    清理

    内部:
      _pending_requests: request_id → asyncio.Future
        用于将引擎输出路由到等待的 generate() 调用
    """

    VLLM_ENGINE_READY_TIMEOUT_S = 30  # 等待引擎就绪的超时时间

    def __init__(
        self,
        vllm_config: EngineConfig,
        engine_count: int = 1,
    ):
        self.vllm_config = vllm_config
        self._engine_count = engine_count

        # ---- 生成 ZMQ 地址 ----
        self._zmq_addresses = get_engine_zmq_addresses()
        self._input_address = self._zmq_addresses["input"]
        self._output_address = self._zmq_addresses["output"]

        # ---- ZMQ 上下文 + sockets ----
        # 使用 zmq.asyncio.Context 以支持 async/await
        self._zmq_context = zmq.asyncio.Context()

        # ROUTER: 发送请求到引擎后端
        #   - 多对一路由: 前端 → 多个引擎(DP)
        #   - 靠 identity 帧区分不同引擎
        self._input_socket = self._zmq_context.socket(zmq.ROUTER)
        self._input_socket.bind(self._input_address)
        logger.info("ROUTER 已绑定: %s", self._input_address)

        # PULL: 接收引擎后端的推理输出
        #   - 公平队列: 多个引擎的输出汇入一个 socket
        self._output_socket = self._zmq_context.socket(zmq.PULL)
        self._output_socket.bind(self._output_address)
        logger.info("PULL 已绑定: %s", self._output_address)

        # ---- 启动引擎进程 ----
        self._engine_manager = CoreEngineProcManager(
            engine_count=engine_count,
            vllm_config=vllm_config,
            input_address=self._input_address,
            output_address=self._output_address,
        )
        self._engine_manager.start_all()

        # ---- 输出路由 ----
        # _pending_requests: request_id → Future
        # generate() 发送请求后创建一个 Future,
        # _process_outputs() 收到输出后 resolve
        # request_id -> (Future, 是否需要返回引擎内部性能指标)。普通在线请求
        # 仍只得到文本；benchmark 请求得到完整 output 字典。
        self._pending_requests: dict[
            str, tuple[asyncio.Future[Any], bool]
        ] = {}

        # ---- 状态 ----
        self._started = False
        self._engine_dead = False
        self._output_handler_task: asyncio.Task | None = None
        self._engine_identities: list[bytes] = []

        logger.info(
            "MPClient 构造完成: ZMQ sockets 已绑定, %d 个引擎进程已启动",
            engine_count,
        )

    @property
    def engine_dead(self) -> bool:
        return self._engine_dead

    # ---- 启动 (异步握手) ----

    async def start(self) -> None:
        """启动传输层 — 等待引擎 READY 握手 + 启动输出处理

        必须在 asyncio event loop 中调用
        """
        if self._started:
            return

        # 等待所有引擎发送 READY
        # ROUTER 收到: [identity, b"READY"]
        logger.info("等待 %d 个引擎就绪...", self._engine_count)
        for i in range(self._engine_count):
            identity, msg = await self._input_socket.recv_multipart()
            assert msg == b"READY", f"期望 READY, 收到 {msg!r}"
            self._engine_identities.append(identity)
            logger.info(
                "引擎 %s 已就绪 (第 %d/%d 个)",
                identity.hex(),
                i + 1,
                self._engine_count,
            )

        # 启动输出处理后台任务
        loop = asyncio.get_running_loop()
        self._output_handler_task = loop.create_task(self._process_outputs())

        self._started = True
        logger.info("MPClient 启动完成: %d 个引擎已就绪", self._engine_count)

    # ---- 输出处理 ----

    async def _process_outputs(self) -> None:
        """后台任务: PULL socket → routing → Future.resolve()

        对应 vLLM AsyncMPClient.process_outputs_socket()

        持续从 PULL 读取引擎输出, 根据 request_id 路由到对应的 Future
        """
        logger.info("输出处理任务已启动")
        try:
            while True:
                output = await self._output_socket.recv_json()
                request_id = output.get("request_id", "")
                logger.debug("收到引擎输出: request_id=%s", request_id)

                pending = self._pending_requests.pop(request_id, None)
                if pending is not None:
                    future, return_metrics = pending
                else:
                    future = None
                    return_metrics = False
                if future is not None and not future.done():
                    future.set_result(output if return_metrics else output["text"])
                else:
                    logger.warning(
                        "收到未知 request_id 的输出: %s", request_id
                    )
        except asyncio.CancelledError:
            logger.info("输出处理任务已取消")
        except Exception:
            logger.exception("输出处理任务异常")

    # ---- 推理接口 ----

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        *,
        return_metrics: bool = False,
        request_start_ns: int | None = None,
        ignore_eos: bool = False,
    ) -> str | dict[str, Any]:
        """发送推理请求到引擎后端, 等待返回结果

        流程:
          1. 生成 request_id, 创建 Future
          2. 通过 ROUTER 发送到引擎
          3. await Future (由 _process_outputs resolve)

        Args:
            prompt: 文本 prompt（后续扩展为完整的 EngineCoreRequest）
            max_tokens: 最大生成 token 数

        Returns:
            引擎生成的文本
        """
        if not self._started:
            raise RuntimeError("MPClient 尚未启动, 请先调用 start()")

        request_id = uuid.uuid4().hex
        request = {
            "request_id": request_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "benchmark_start_ns": request_start_ns,
            "ignore_eos": ignore_eos,
        }

        # 创建 Future 用于等待结果
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending_requests[request_id] = (future, return_metrics)

        # 通过 ROUTER 发送到引擎（发送给第一个引擎）
        # ROUTER.send_multipart([identity, json_data])
        #   → DEALER 收到: [json_data]  (identity 被自动 strip)
        identity = self._engine_identities[0]
        await self._input_socket.send_multipart(
            [identity, json.dumps(request).encode()]
        )
        logger.info(
            "已发送请求: request_id=%s, prompt=%.40s...", request_id, prompt
        )

        # 等待引擎返回结果
        return await future

    # ---- 关闭 ----

    def shutdown(self) -> None:
        """关闭传输层 — 停止引擎进程, 清理 ZMQ 资源"""
        logger.info("MPClient 正在关闭...")

        # 取消输出处理任务
        if self._output_handler_task is not None:
            self._output_handler_task.cancel()

        # 终止引擎进程
        shutdown_processes(self._engine_manager.processes)

        # 关闭 ZMQ sockets
        if hasattr(self, "_input_socket"):
            self._input_socket.close()
        if hasattr(self, "_output_socket"):
            self._output_socket.close()
        if hasattr(self, "_zmq_context"):
            self._zmq_context.term()

        logger.info("MPClient 已关闭")
