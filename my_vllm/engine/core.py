"""
引擎后端 — EngineCore + EngineCoreProc

分层：
  EngineCore:        纯推理逻辑（scheduler + executor + KV cache）
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
from collections import deque

import zmq

from my_vllm.config import EngineConfig

logger = logging.getLogger(__name__)


class EngineCore:
    """引擎后端基类 — 纯推理逻辑，不含通信

    一个 EngineCore = 一个完整的调度域（一个 scheduler + 一组 worker）
    对应 vLLM 的 EngineCore

    组件：
      - self.scheduler:        调度器（请求队列 + KV cache 分配 + 输出推进）
      - self.kv_cache_manager: KV cache 管理器（block 池 + 前缀缓存 + 引用计数）
      - self.model_executor:   模型执行器（管理 GPU worker，执行 forward）
      - self.kv_cache_config:  Worker 显存布局 + KV cache group 形状配置
    """

    def __init__(self, vllm_config: EngineConfig):
        self.vllm_config = vllm_config
        self._is_running = True
        self.tokenizer = None
        if vllm_config.model != "test-model":
            from my_vllm.tokenizer import HuggingFaceTokenizer

            self.tokenizer = HuggingFaceTokenizer(vllm_config.model)

        # 创建执行器：拉起所有 worker 进程。
        # worker 的【阶段 1/3】init_device 与【阶段 2/3】load_model 在
        # worker 进程内直接执行，此调用返回时 worker 已就绪。
        from my_vllm.executor.multiproc_executor import MultiprocExecutor

        self.model_executor = MultiprocExecutor(vllm_config)

        # Worker 先根据真实模型给出 spec 并 profiling；EngineCore 据此生成唯一的
        # KVCacheConfig，再同时交给 GPU 物理张量和 CPU BlockPool 使用。
        self.kv_cache_config = self._initialize_kv_caches()

        from my_vllm.v1.core.kv_cache_manager import KVCacheManager

        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=self.kv_cache_config,
            max_model_len=vllm_config.max_model_len,
            enable_caching=vllm_config.enable_prefix_caching,
        )

        from my_vllm.v1.core.sched.scheduler import Scheduler

        self.scheduler = Scheduler(vllm_config, self.kv_cache_manager)
        self.batch_queue_size = (
            vllm_config.parallel_config.pipeline_parallel_size
            if vllm_config.parallel_config.pipeline_parallel_size > 1
            else 1
        )
        self.batch_queue = deque(maxlen=self.batch_queue_size)

        logger.info(
            "EngineCore 初始化完成 (model=%s, max_model_len=%d)",
            vllm_config.model,
            vllm_config.max_model_len,
        )

    def _initialize_kv_caches(self):
        """【Worker 初始化 · 阶段 3/3】Initialize KV Cache

        对应 vLLM EngineCore._initialize_kv_caches()。

        阶段 1（init_device）和阶段 2（load_model）在 worker 进程内直接执行；
        阶段 3 由 EngineCore 通过 collective_rpc 统一触发，让每个 worker 分配
        KV cache 显存。
        """
        from my_vllm.v1.kv_cache_interface import generate_kv_cache_config

        worker_specs = self.model_executor.get_kv_cache_specs()
        model_memory_per_worker = self.model_executor.get_model_memory_usage()
        available_per_worker = self.model_executor.determine_available_memory()
        # 【PP KV 配置】不同 stage 拥有不同 layer_names，不能把 rank0 的配置
        # 原样广播给所有 worker。先按每个 rank 的本地层数/页大小求容量，再取
        # 全局最小 block 数，使 Scheduler 的一个逻辑 block id 在所有 stage
        # 都有对应物理页。
        capacities = [
            available // sum(spec.page_size_bytes for spec in specs.values())
            for available, specs in zip(
                available_per_worker, worker_specs, strict=True
            )
        ]
        common_num_blocks = (
            self.vllm_config.num_gpu_blocks
            if self.vllm_config.num_gpu_blocks is not None
            else min(capacities)
        )
        worker_configs = [
            generate_kv_cache_config(
                specs,
                available,
                num_blocks_override=common_num_blocks,
            )
            for specs, available in zip(
                worker_specs, available_per_worker, strict=True
            )
        ]
        results = self.model_executor.initialize_from_config(worker_configs)
        # Scheduler 只消费 group/block_size/num_blocks；单一 FullAttention group
        # 下任选一个 rank 的本地配置即可，layer_names 仅供对应 Worker 绑定。
        kv_cache_config = worker_configs[0]
        logger.info(
            "KV cache 初始化完成：model_memory_per_worker=%s, "
            "kv_layers_per_worker=%s, bytes_per_block_per_worker=%s, "
            "available_per_worker=%s, common_blocks=%d, ack=%s",
            model_memory_per_worker,
            [len(specs) for specs in worker_specs],
            [
                sum(spec.page_size_bytes for spec in specs.values())
                for specs in worker_specs
            ],
            available_per_worker,
            kv_cache_config.num_blocks,
            results,
        )
        return kv_cache_config

    def is_running(self) -> bool:
        return self._is_running

    def shutdown(self) -> None:
        """关闭引擎，清理资源（含 worker 进程）"""
        self._is_running = False
        if hasattr(self, "model_executor"):
            self.model_executor.shutdown()
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
            1. _process_input_queue()  — 把 input_queue 里的请求转成 Request 交给 scheduler
            2. scheduler.schedule()    — 选出本轮要算的请求 + 分配 KV cache
            3. executor.execute_model()— collective_rpc 广播给 worker 执行前向 + 采样
            4. scheduler.update_from_output() — 推进请求状态 / 判定结束
            5. _send_finished_outputs()       — 把已结束请求的结果送回前端
        """
        logger.info(
            "EngineCore 进入主循环 (index=%d, dp_size=%d)",
            self.engine_index,
            self.vllm_config.data_parallel_size,
        )

        while self.is_running():
            # 1) 批量接收新请求，交给调度器
            self._process_input_queue()

            if self.batch_queue_size > 1:
                self._step_with_batch_queue()
                time.sleep(0.001)
                continue

            # 2) 同步路径：execute_model 与 sample_tokens 仍是两条独立 RPC。
            scheduler_output = self.scheduler.schedule()

            # 3) 执行：即使是 PP=1，也保留 ModelRunner V2 的两段式接口：
            #    execute_model 只前向，sample_tokens 独立完成 logits/采样。
            # 即使本轮没有 token，只要有 finished 通知也必须 RPC 一次，让
            # Worker 清掉 CachedRequestState/InputBatch 槽位。
            if (
                scheduler_output.total_num_scheduled_tokens > 0
                or scheduler_output.finished_req_ids
            ):
                self.model_executor.execute_model(scheduler_output)
                model_runner_output = self.model_executor.sample_tokens()
                # 调试日志：打印本轮 collective_rpc 回传的采样结果，直观确认执行器链路生效
                sampled_text = {
                    rid: (
                        self.tokenizer.decode(ids)
                        if self.tokenizer is not None
                        else "".join(chr(t) for t in ids)
                    )
                    for rid, ids in zip(
                        model_runner_output.req_ids,
                        model_runner_output.sampled_token_ids,
                    )
                }
                logger.info("[RPC] execute_model 本轮采样结果: %s", sampled_text)
                if scheduler_output.total_num_scheduled_tokens > 0:
                    self.scheduler.update_from_output(
                        scheduler_output, model_runner_output
                    )

            # 4) 回传已结束请求的输出
            self._send_finished_outputs()

            # 空闲时小睡，避免忙等占满 CPU
            time.sleep(0.005)

        logger.info("EngineCore 退出主循环 (index=%d)", self.engine_index)

    def _scheduler_has_work(self) -> bool:
        return bool(
            self.scheduler.waiting
            or self.scheduler._finished_req_ids_to_notify
            or any(
                request.num_tokens > request.num_computed_tokens
                for request in self.scheduler.running
            )
        )

    def _step_with_batch_queue(self) -> None:
        """【异步 PP】优先填充 batch queue，满队列后核销最老采样 Future。"""

        submitted = False
        if len(self.batch_queue) < self.batch_queue_size and self._scheduler_has_work():
            scheduler_output = self.scheduler.schedule()
            if (
                scheduler_output.total_num_scheduled_tokens > 0
                or scheduler_output.finished_req_ids
            ):
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
                sample_future = self.model_executor.sample_tokens(non_block=True)
                self.batch_queue.appendleft(
                    (sample_future, scheduler_output, exec_future)
                )
                submitted = True

        # 队列尚未填满且仍有新批可发：先制造 PP 在途批次，不在这里阻塞。
        if (
            submitted
            and len(self.batch_queue) < self.batch_queue_size
            and self._scheduler_has_work()
        ):
            return
        if not self.batch_queue:
            self._send_finished_outputs()
            return

        sample_future, scheduler_output, exec_future = self.batch_queue.pop()
        model_runner_output = sample_future.result()
        # execute 返回 None 是正常语义；result() 在失败时会重新抛出 worker 异常。
        exec_future.result()
        if scheduler_output.total_num_scheduled_tokens > 0:
            self.scheduler.update_from_output(
                scheduler_output, model_runner_output
            )
        self._send_finished_outputs()

    # ---- 主循环内部步骤 ----

    def _process_input_queue(self) -> None:
        """把 input_queue 里的原始请求全部取出，转成 Request 交给调度器"""
        while True:
            try:
                raw = self.input_queue.get_nowait()
            except queue.Empty:
                break
            request = self._build_request(raw)
            self.scheduler.add_request(request)

    def _build_request(self, raw: dict) -> "Request":
        """把前端传来的 {request_id, prompt} 转成 Request 对象

        test-model 使用字符级编码；真实模型使用模型目录里的 tokenizer.json。
        """
        from my_vllm.v1.request import Request, SamplingParams

        prompt = raw.get("prompt", "")
        prompt_token_ids = (
            self.tokenizer.encode(prompt)
            if self.tokenizer is not None
            else [ord(ch) for ch in prompt]
        )
        max_tokens = raw.get("max_tokens", self.vllm_config.max_model_len)
        return Request(
            request_id=raw["request_id"],
            prompt_token_ids=prompt_token_ids,
            sampling_params=SamplingParams(
                max_tokens=max_tokens,
                stop_token_ids=(
                    self.tokenizer.eos_token_ids
                    if self.tokenizer is not None
                    else ()
                ),
            ),
        )

    def _send_finished_outputs(self) -> None:
        """把本步判定结束的请求结果送回 output_queue → 前端

        finish_reason 取 Request.get_finished_reason()；detokenize 用字符级解码
        （与 _build_request 的字符级编码对应，真实实现替换为 tokenizer.decode()）。
        """
        for req_id in list(self.scheduler.finished_req_ids):
            request = self.scheduler.requests.get(req_id)
            self.scheduler.finished_req_ids.discard(req_id)
            if request is None:
                continue

            finish_reason = request.get_finished_reason()
            output = {
                "request_id": req_id,
                "text": self._detokenize(request.output_token_ids),
                "finish_reason": finish_reason.value if finish_reason else "stop",
            }
            self.output_queue.put(output)
            self.scheduler.requests.pop(req_id, None)
            logger.info(
                "请求 %s 完成: finish_reason=%s, output_tokens=%d",
                req_id, finish_reason, request.num_output_tokens,
            )

    def _detokenize(self, token_ids: list[int]) -> str:
        if self.tokenizer is not None:
            return self.tokenizer.decode(token_ids)
        return "".join(chr(t) for t in token_ids)

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
