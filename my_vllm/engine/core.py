"""
引擎后端 — EngineCore + EngineCoreProc

分层：
  EngineCore:        纯推理逻辑（scheduler + executor + KV cache）— 预留
  EngineCoreProc:    子进程包装 + ZMQ 通信层（DEALER/PUSH + IO 线程）

对应 vLLM 的 vllm/v1/engine/core.py
"""

import json
import logging
import os
import queue
import signal
import threading
import time

import zmq

from my_vllm.config import EngineConfig

logger = logging.getLogger(__name__)


class EngineCore:
    """引擎后端基类 — 纯推理逻辑，不含通信

    一个 EngineCore = 一个完整的调度域（一个 scheduler + 一组 worker）
    对应 vLLM 的 EngineCore

    预留组件：
      - self.scheduler:       调度器（管理请求队列 + KV cache 分配）
      - self.model_executor:  模型执行器（管理 GPU worker，执行 forward）
      - self.kv_cache_config: KV cache 配置（block 大小、数量等）
    """

    def __init__(self, vllm_config: EngineConfig):
        self.vllm_config = vllm_config
        self._is_running = True
        logger.info(
            "EngineCore 初始化完成 (model=%s, max_model_len=%d)",
            vllm_config.model,
            vllm_config.max_model_len,
        )

    def is_running(self) -> bool:
        return self._is_running

    def shutdown(self) -> None:
        """关闭引擎，清理资源"""
        self._is_running = False
        logger.info("EngineCore 已关闭")


class EngineCoreProc(EngineCore):
    """引擎后端进程类 — 子进程入口 + ZMQ 通信包装

    对应 vLLM 的 EngineCoreProc(EngineCore)

    ZMQ 通信架构:
      DEALER (connect) ← ROUTER (bind) 前端发送请求
      PUSH   (connect) → PULL   (bind) 前端接收输出

    内部队列:
      input_queue  (Queue):  ZMQ 输入线程 → 主循环
      output_queue (Queue):  主循环 → ZMQ 输出线程
    """

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"

    # ==================================================================
    # 子进程入口
    # ==================================================================

    @staticmethod
    def run_engine_core(
        vllm_config: EngineConfig,
        input_address: str,
        output_address: str,
        engine_index: int = 0,
    ):
        """子进程入口点 — multiprocessing.Process 的 target

        对应 vLLM EngineCoreProc.run_engine_core():
          1. 构造 EngineCoreProc 实例 → __init__ (握手 + 创建 ZMQ + IO 线程)
          2. 注册 SIGTERM/SIGINT 信号处理
          3. 调用 engine_core.run_busy_loop() → 进入主循环
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

        engine_core: "EngineCoreProc | None" = None
        try:
            logger.info(
                "EngineCore 子进程启动 (index=%d, pid=%d)", engine_index, os.getpid()
            )
            logger.info(
                "ZMQ 地址: input=%s, output=%s", input_address, output_address
            )

            engine_core = EngineCoreProc(
                vllm_config,
                input_address=input_address,
                output_address=output_address,
                engine_index=engine_index,
            )

            def signal_handler(signum, frame):
                sig_name = signal.Signals(signum).name
                logger.info(
                    "[shutdown] EngineCore 收到信号 %s, 准备退出", sig_name
                )
                engine_core._is_running = False

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.info("[shutdown] EngineCore 主循环退出")
            raise
        except Exception:
            logger.exception("EngineCore 异常退出")
            raise
        finally:
            if engine_core is not None:
                engine_core.shutdown()
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)

    # ==================================================================
    # 实例方法
    # ==================================================================

    def __init__(
        self,
        vllm_config: EngineConfig,
        input_address: str,
        output_address: str,
        engine_index: int = 0,
    ):
        super().__init__(vllm_config)
        self.engine_index = engine_index

        # ---- 内部队列 (线程安全) ----
        # input_queue:  ZMQ 输入线程把收到的请求放这里
        # output_queue: 主循环把推理结果放这里，ZMQ 输出线程取走
        self.input_queue: queue.Queue[dict] = queue.Queue()
        self.output_queue: queue.Queue[dict] = queue.Queue()

        # ---- ZMQ 通信层 ----
        # 每个引擎进程有自己的 ZMQ Context
        self._zmq_context = zmq.Context()

        # DEALER: 异步接收来自前端的请求
        # 对应前端 ROUTER socket
        self._dealer = self._zmq_context.socket(zmq.DEALER)
        identity = engine_index.to_bytes(length=2, byteorder="little")
        self._dealer.setsockopt(zmq.IDENTITY, identity)
        self._dealer.connect(input_address)
        logger.info("DEALER 已连接到 %s (identity=%s)", input_address, identity.hex())

        # PUSH: 异步发送推理结果回前端
        # 对应前端 PULL socket
        self._push = self._zmq_context.socket(zmq.PUSH)
        self._push.connect(output_address)
        logger.info("PUSH 已连接到 %s", output_address)

        # ---- 握手: 向前端发送 READY 信号 ----
        self._dealer.send(b"READY")
        logger.info(
            "已向前端发送 READY 信号 (identity=%s)", identity.hex()
        )

        # ---- 启动 IO 线程 ----
        # input_thread:  ZMQ DEALER → input_queue
        # output_thread: output_queue → ZMQ PUSH
        self._input_thread = threading.Thread(
            target=self._process_input_socket,
            name=f"engine_input_{engine_index}",
            daemon=True,
        )
        self._input_thread.start()

        self._output_thread = threading.Thread(
            target=self._process_output_socket,
            name=f"engine_output_{engine_index}",
            daemon=True,
        )
        self._output_thread.start()

        logger.info(
            "EngineCoreProc 初始化完成 (index=%d, pid=%d, identity=%s)",
            engine_index,
            os.getpid(),
            identity.hex(),
        )

    # ---- IO 线程 ----

    def _process_input_socket(self) -> None:
        """输入线程 — 从 ZMQ DEALER 读取请求, 放入 input_queue

        对应 vLLM EngineCoreProc.process_input_sockets()

        使用 Poller + timeout 而非阻塞 recv_multipart(),
        以便定期检查 is_running() 从而支持优雅关闭。
        """
        logger.info("输入线程已启动")
        poller = zmq.Poller()
        poller.register(self._dealer, zmq.POLLIN)

        while self.is_running():
            try:
                # 用 poller 实现可中断的等待（500ms 超时, 用于检查 is_running）
                socks = dict(poller.poll(timeout=500))
                if self._dealer not in socks:
                    continue

                # DEALER 收到的消息: 前端 ROUTER 发送的 [identity, json_data]
                #   → DEALER 自动去掉 identity, 只收到 [json_data]
                frames = self._dealer.recv_multipart()
                if not frames:
                    continue
                if frames[0] == self.ENGINE_CORE_DEAD:
                    continue

                request = json.loads(frames[-1])
                logger.debug(
                    "收到请求: request_id=%s", request.get("request_id")
                )
                self.input_queue.put(request)

            except zmq.ZMQError:
                if self.is_running():
                    logger.warning("输入线程 ZMQ 错误", exc_info=True)
                break
            except Exception:
                logger.exception("输入线程异常")
        logger.info("输入线程已退出")

    def _process_output_socket(self) -> None:
        """输出线程 — 从 output_queue 读取结果, 通过 ZMQ PUSH 发送

        对应 vLLM EngineCoreProc.process_output_sockets()
        """
        logger.info("输出线程已启动")
        while self.is_running():
            try:
                output = self.output_queue.get(timeout=0.5)
                self._push.send_json(output)
                logger.debug(
                    "已发送输出: request_id=%s", output.get("request_id")
                )
            except queue.Empty:
                continue
            except zmq.ZMQError:
                if self.is_running():
                    logger.warning("输出线程 ZMQ 错误", exc_info=True)
                break
            except Exception:
                logger.exception("输出线程异常")
        logger.info("输出线程已退出")

    # ---- 主循环 ----

    def run_busy_loop(self) -> None:
        """主循环 — 引擎后端的核心工作循环

        对应 vLLM EngineCore.run_busy_loop():
          while is_running:
            1. _process_input_queue()  — 从 input_queue 取 EngineCoreRequest
            2. _process_engine_step()  — 调用 scheduler.schedule() + executor 前向

        当前阶段: 从 input_queue 取请求, 生成模拟结果, 放入 output_queue
        后续 commit 实现真实的 scheduler + executor
        """
        logger.info(
            "EngineCore 进入主循环 (index=%d, dp_size=%d)",
            self.engine_index,
            self.vllm_config.data_parallel_size,
        )

        while self.is_running():
            # 1) 从 input_queue 取请求（带超时, 以便检查 is_running）
            try:
                request = self.input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            logger.info(
                "EngineCore 处理请求: request_id=%s, prompt=%.40s...",
                request.get("request_id"),
                request.get("prompt", ""),
            )

            # 2) TODO: 调用 scheduler + executor 执行真实推理
            #    self._process_engine_step()

            # 模拟推理延迟
            time.sleep(0.2)

            # 3) 构造输出, 放入 output_queue → 输出线程发送回前端
            output = {
                "request_id": request["request_id"],
                "text": (
                    f"[引擎回复 index={self.engine_index}] "
                    f"收到: {request.get('prompt', '')[:50]}..."
                ),
                "finish_reason": "stop",
            }
            self.output_queue.put(output)

        logger.info("EngineCore 退出主循环 (index=%d)", self.engine_index)

    def shutdown(self) -> None:
        """关闭引擎, 清理 ZMQ 资源"""
        super().shutdown()
        if hasattr(self, "_dealer"):
            self._dealer.close()
        if hasattr(self, "_push"):
            self._push.close()
        if hasattr(self, "_zmq_context"):
            self._zmq_context.term()
        logger.info("EngineCoreProc ZMQ 资源已清理")
