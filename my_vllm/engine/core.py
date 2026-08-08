"""
引擎后端 — EngineCore + EngineCoreProc

分层：
  EngineCore:        纯推理逻辑（scheduler + executor + KV cache）— 预留
  EngineCoreProc:    子进程包装 + ZMQ 通信层 — 预留

对应 vLLM 的 vllm/v1/engine/core.py
"""

import logging
import os
import signal
import time

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

    对应 vLLM 的 EngineCoreProc(EngineCore)：
      - run_engine_core()  静态方法，multiprocessing.Process 的 target
      - __init__()         构造 ZMQ sockets + IO 线程（预留）
      - run_busy_loop()   主循环：读 input_queue → 调度 → 执行 → 写 output_queue
    """

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"

    # ---- 子进程入口 ----

    @staticmethod
    def run_engine_core(
        vllm_config: EngineConfig,
        engine_index: int = 0,
    ):
        """子进程入口点 — multiprocessing.Process 的 target

        对应 vLLM EngineCoreProc.run_engine_core():
          1. 构造 EngineCoreProc 实例 → __init__ (握手 + 创建线程)
          2. 注册 SIGTERM/SIGINT 信号处理
          3. 调用 engine_core.run_busy_loop() → 进入主循环
        """
        # 子进程内配置日志（因为 fork 后不会继承父进程的 logging 配置）
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
            engine_core = EngineCoreProc(vllm_config, engine_index=engine_index)

            # 注册信号处理 — 优雅关闭
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

    # ---- 实例方法 ----

    def __init__(self, vllm_config: EngineConfig, engine_index: int = 0):
        super().__init__(vllm_config)
        self.engine_index = engine_index

        # 预留：input_queue（ZMQ 输入线程 → 主循环）
        # 预留：output_queue（主循环 → ZMQ 输出线程）
        # 预留：ZMQ sockets (DEALER/PUSH)
        # 预留：input_thread / output_thread

        logger.info(
            "EngineCoreProc 初始化完成 (index=%d, pid=%d)",
            engine_index,
            os.getpid(),
        )

    def run_busy_loop(self) -> None:
        """主循环 — 引擎后端的核心工作循环

        对应 vLLM EngineCore.run_busy_loop():
          while is_running:
            1. 从 input_queue 取 EngineCoreRequest
            2. 调用 scheduler.schedule() 分配 KV cache + 选取 batch
            3. 调用 model_executor.execute_model() 执行 forward
            4. 将 EngineCoreOutput 放入 output_queue

        当前为占位实现，仅打印心跳日志。
        """
        logger.info(
            "EngineCore 进入主循环 (index=%d, dp_size=%d)",
            self.engine_index,
            self.vllm_config.data_parallel_size,
        )
        heartbeat_count = 0
        while self.is_running():
            # TODO: _process_input_queue()   — 从 input_queue 取请求
            # TODO: _process_engine_step()  — 调用 scheduler + executor

            time.sleep(5)
            heartbeat_count += 1
            logger.info(
                "EngineCore 心跳 #%d (index=%d) — 预留：等待 ZMQ 通信接入",
                heartbeat_count,
                self.engine_index,
            )

        logger.info("EngineCore 退出主循环 (index=%d)", self.engine_index)
