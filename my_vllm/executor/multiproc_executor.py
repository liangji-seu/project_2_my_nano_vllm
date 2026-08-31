"""
多进程执行器 — 拉起 Worker 进程 + 有序异步 collective_rpc

对应 vLLM 的 vllm/v1/executor/multiproc_executor.py

架构（简化版，单机多卡）：
  MultiprocExecutor（在 EngineCore 进程内）
    ├─ rpc_broadcast_mq（writer）      → 广播 RPC 给所有 worker
    ├─ WorkerProc #0（独立进程）         ──┐
    ├─ WorkerProc #1（独立进程）           ├─ 每个 worker 有自己的 response_mq
    └─ ...                              ──┘
    └─ response_mqs[rank]（reader）    ← 收集每个 worker 的回复

简化说明（相比 vLLM 源码）：
  - collective_rpc 同时支持同步等待和 ``non_block=True`` 异步提交。
  - 异步回复由单线程严格按 RPC 广播顺序回收，避免相邻 micro-batch
    的 response MQ 消息被不同 Future 交叉消费。
  - 省略多机 TP/PP（peer_worker_response_mqs）、DP、worker 健康监控线程等。
  - 省略 cloudpickle 字节方法分支（RPC 方法统一用字符串名）。
"""

import logging
import signal
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from multiprocessing import Process, get_context
from multiprocessing.connection import Connection, wait as connection_wait

import torch

from my_vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from my_vllm.distributed.utils import get_open_port

logger = logging.getLogger(__name__)


@dataclass
class UnreadyWorkerProcHandle:
    """尚未就绪（READY）的 worker 进程句柄

    对应 vLLM 的 UnreadyWorkerProcHandle。
    """
    proc: Process
    rank: int
    ready_pipe: Connection            # 子进程 → 父进程：报告 READY + 回复队列地址
    death_writer: Connection | None = None  # 父进程持有：关闭它通知子进程父已退出


@dataclass
class WorkerProcHandle:
    """一个已就绪（READY）的 worker 进程句柄

    对应 vLLM 的 WorkerProcHandle。
    """
    proc: Process
    rank: int
    worker_response_mq: MessageQueue  # executor 从这个队列读该 worker 的回复
    death_writer: Connection | None = None


class WorkerProc:
    """单个 worker 进程的完整生命周期管理

    对应 vLLM 的 WorkerProc。

    在子进程中执行：
      1. 构造 WorkerWrapperBase + Worker（init_worker）
      2. 【阶段 1/3】init_device
      3. 【阶段 2/3】load_model
      4. 建立 IPC 通信（rpc_broadcast_mq reader + worker_response_mq writer）
      5. 发送 READY + 握手，然后进入 worker_busy_loop
    """

    READY_STR = "READY"

    class ResponseStatus(Enum):
        """worker 回复的封装状态"""
        SUCCESS = auto()
        FAILURE = auto()

    def __init__(
        self,
        vllm_config,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        rpc_broadcast_address: str,
    ):
        self.rank = rank
        self.local_rank = local_rank

        # 1. 构造 WorkerWrapperBase（适配转发层），再在其内部构造真正的 Worker
        from my_vllm.worker.worker_base import WorkerWrapperBase

        wrapper = WorkerWrapperBase(rpc_rank=local_rank, global_rank=rank)

        # all_kwargs 长度为 world_size，本进程只填自己 local_rank 那份
        # （vLLM 传 all_kwargs 是为了支持多 executor 共享 worker，单机下只有一份有值）
        all_kwargs: list[dict] = [{} for _ in range(vllm_config.parallel_config.world_size)]
        all_kwargs[local_rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": (rank % vllm_config.parallel_config.tensor_parallel_size == 0),
        }
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper

        # 2. 【阶段 1/3】init_device + 【阶段 2/3】load_model（在 worker 进程内直接执行）
        self.worker.init_device()
        self.worker.load_model()

        # 3. 建立 IPC 通信
        # 接收 executor 广播的队列（本进程是 reader）
        self.rpc_broadcast_mq = MessageQueue(
            n_reader=vllm_config.parallel_config.world_size,
            is_writer=False,
            address=rpc_broadcast_address,
        )
        # 回传结果给 executor 的队列（本进程是 writer，executor 是唯一 reader）
        self.worker_response_mq = MessageQueue(n_reader=1, is_writer=True)

        logger.info(
            "WorkerProc rank=%d 构造完成 (device 与模型已初始化)",
            rank,
        )

    # ---- 输出打包 + 主循环 ----

    def enqueue_output(self, output) -> None:
        """把 worker 的输出打包成 (SUCCESS/FAILURE, result) 回传给 executor

        对应 vLLM 的 enqueue_output()。
        若 output 是 Exception，转成 FAILURE（异常对象往往不可序列化，转字符串）。
        """
        if isinstance(output, Exception):
            result = (self.ResponseStatus.FAILURE, str(output))
        else:
            result = (self.ResponseStatus.SUCCESS, output)
        self.worker_response_mq.enqueue(result)

    def worker_busy_loop(self, shutdown_requested: threading.Event) -> None:
        """worker 进程的 RPC 服务主循环

        对应 vLLM 的 worker_busy_loop()。

        不断从广播队列取 RPC 请求 → 调 worker 对应方法 → 把结果回传。
        用 poll 超时而不是无限阻塞，是为了能定期检查 shutdown_requested 从而优雅退出。
        """
        while not shutdown_requested.is_set():
            try:
                method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(timeout=0.5)
            except TimeoutError:
                continue

            try:
                # method 是字符串，用 getattr 找到 worker 的方法（wrapper 会转发给真正 worker）
                func = getattr(self.worker, method)
                output = func(*args, **kwargs)
                # output_rank 为 None 表示所有 worker 都回传；否则只有指定 rank 回传
                if output_rank is None or self.rank == output_rank:
                    self.enqueue_output(output)
            except Exception as e:  # noqa: BLE001 —— 主循环必须兜底，避免进程崩溃
                logger.exception("WorkerProc rank=%d 执行 %s 出错", self.rank, method)
                if output_rank is None or self.rank == output_rank:
                    self.enqueue_output(e)

        logger.info("WorkerProc rank=%d 退出主循环", self.rank)

    def monitor_death_pipe(
        self,
        death_pipe: Connection | None,
        shutdown_requested: threading.Event,
    ) -> None:
        """启动后台线程监听父进程存活

        对应 vLLM 的 monitor_death_pipe()。
        父进程退出 → 管道关闭 → EOFError → 置 shutdown_requested，让主循环优雅退出。
        """
        if death_pipe is None:
            return

        def _monitor() -> None:
            try:
                death_pipe.recv()  # 阻塞直到父进程退出（管道关闭 → EOFError）
            except EOFError:
                logger.info("父进程退出，worker rank=%d 准备关闭", self.rank)
                shutdown_requested.set()
            except Exception:
                pass

        threading.Thread(target=_monitor, daemon=True, name="DeathPipeMonitor").start()

    def shutdown(self) -> None:
        """关闭消息队列 + worker"""
        if self.rpc_broadcast_mq is not None:
            self.rpc_broadcast_mq.shutdown()
            self.rpc_broadcast_mq = None
        if self.worker_response_mq is not None:
            self.worker_response_mq.shutdown()
            self.worker_response_mq = None
        self.worker.shutdown()

    # ---- 工厂方法（在 executor 进程内调用）----

    @staticmethod
    def make_worker_process(
        vllm_config,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        rpc_broadcast_address: str,
    ) -> UnreadyWorkerProcHandle:
        """创建一个 worker 子进程（未就绪），返回句柄

        对应 vLLM 的 make_worker_process()。

        两根管道：
          ready_pipe  — 子进程 → 父进程：报告 READY + 回复队列地址
          death_pipe  — 父进程持有写端，关闭时子进程读端 EOF，据此检测父进程退出
        """
        context = get_context("spawn")  # 统一用 spawn，保证子进程干净启动

        ready_reader, ready_writer = context.Pipe(duplex=False)
        death_reader, death_writer = context.Pipe(duplex=False)

        proc = context.Process(
            target=WorkerProc.worker_main,
            kwargs={
                "vllm_config": vllm_config,
                "local_rank": local_rank,
                "rank": rank,
                "distributed_init_method": distributed_init_method,
                "rpc_broadcast_address": rpc_broadcast_address,
                "ready_pipe": ready_writer,
                "death_pipe": death_reader,
            },
            name=f"Worker_{rank}",
            daemon=True,
        )
        proc.start()

        # 父进程关闭自己用不到的管道端
        ready_writer.close()
        death_reader.close()
        # 保留 death_writer：父进程退出时会关闭它，子进程据此感知
        return UnreadyWorkerProcHandle(proc, rank, ready_reader, death_writer)

    @staticmethod
    def wait_for_ready(
        unready_handles: list[UnreadyWorkerProcHandle],
    ) -> list[WorkerProcHandle]:
        """阻塞等待所有 worker 就绪，返回已就绪句柄列表（按 rank 排序）

        对应 vLLM 的 wait_for_ready()。
        """
        ready_handles: list[WorkerProcHandle | None] = [None] * len(unready_handles)
        pipes = {h.ready_pipe: h for h in unready_handles}

        while pipes:
            for pipe in connection_wait(list(pipes.keys())):
                h = pipes.pop(pipe)
                try:
                    response = pipe.recv()
                    assert response["status"] == WorkerProc.READY_STR, (
                        f"worker rank={h.rank} 未就绪：{response}"
                    )
                    # executor 根据 worker 回传的地址，创建 reader 端 response_mq
                    worker_response_mq = MessageQueue(
                        n_reader=1,
                        is_writer=False,
                        address=response["response_address"],
                    )
                    ready_handles[h.rank] = WorkerProcHandle(
                        proc=h.proc,
                        rank=h.rank,
                        worker_response_mq=worker_response_mq,
                        death_writer=h.death_writer,
                    )
                except EOFError:
                    raise RuntimeError(
                        f"worker rank={h.rank} 初始化失败（进程异常退出）"
                    ) from None
                finally:
                    pipe.close()

        return ready_handles  # type: ignore[return-value]

    @staticmethod
    def worker_main(**kwargs) -> None:
        """worker 子进程的入口函数（multiprocessing.Process 的 target）

        对应 vLLM 的 worker_main()。
        """
        shutdown_requested = threading.Event()

        def _signal_handler(signum, frame):
            # SIGTERM/SIGINT → 置标志并抛 SystemExit 优雅退出
            if not shutdown_requested.is_set():
                shutdown_requested.set()
                raise SystemExit()

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

        ready_pipe = kwargs.pop("ready_pipe")
        death_pipe = kwargs.pop("death_pipe", None)
        rank = kwargs.get("rank", 0)

        worker: WorkerProc | None = None
        try:
            worker = WorkerProc(**kwargs)

            # 监听父进程存活
            worker.monitor_death_pipe(death_pipe, shutdown_requested)

            # 发送 READY + 回复队列地址给 executor
            ready_pipe.send(
                {
                    "status": WorkerProc.READY_STR,
                    "response_address": worker.worker_response_mq.address,
                }
            )

            # 握手（顺序必须与 executor 保持一致，否则死锁）
            worker.rpc_broadcast_mq.wait_until_ready()   # reader：收 READY
            worker.worker_response_mq.wait_until_ready()  # writer：收订阅 → 发 READY

            ready_pipe.close()
            ready_pipe = None

            # 进入 RPC 服务主循环
            worker.worker_busy_loop(shutdown_requested)

        except SystemExit:
            logger.info("Worker rank=%d 收到退出信号", rank)
            raise
        except Exception:
            logger.exception("WorkerProc rank=%d 启动/运行失败", rank)
        finally:
            if ready_pipe is not None:
                ready_pipe.close()
            if death_pipe is not None:
                death_pipe.close()
            if worker is not None:
                worker.shutdown()


class MultiprocExecutor:
    """多进程执行器 — 管理一组 worker 进程，提供 collective_rpc

    对应 vLLM 的 MultiprocExecutor。
    """

    def __init__(self, vllm_config):
        self.vllm_config = vllm_config
        self.parallel_config = vllm_config.parallel_config
        self.world_size = self.parallel_config.world_size

        # 单机多卡（当前只跑一个 DP 副本、无多节点）：本机 worker 数 == world_size
        self.local_world_size = self.world_size

        # 校验 world_size 不超过本机实际 GPU 数；无 CUDA 时跳过（走 CPU 回退）
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if self.world_size > num_gpus:
                raise RuntimeError(
                    f"world_size={self.world_size} "
                    f"(tp={self.parallel_config.tensor_parallel_size} × "
                    f"pp={self.parallel_config.pipeline_parallel_size}) "
                    f"超过本机 GPU 数 {num_gpus}，请调低 --tensor-parallel-size / "
                    f"--pipeline-parallel-size"
                )

        self.workers: list[WorkerProcHandle] = []
        self.response_mqs: list[MessageQueue] = []
        self.rpc_broadcast_mq: MessageQueue | None = None
        # 单 reader 线程严格按 RPC 提交顺序消费 response MQ，避免多个 Future
        # 并发 dequeue 时把 batch A/B 的结果配错。
        self._response_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="my_vllm_rpc_response"
        )

        self._init_executor()

    def _init_executor(self) -> None:
        """拉起所有 worker 进程 + 建立 IPC + 握手

        对应 vLLM MultiprocExecutor._init_executor()。
        """
        # 1. NCCL 握手地址（单机用 tcp://127.0.0.1:<空闲端口>）
        distributed_init_method = f"tcp://127.0.0.1:{get_open_port()}"

        # 2. 广播队列（executor 是 writer，所有 worker 是 reader）
        self.rpc_broadcast_mq = MessageQueue(
            n_reader=self.world_size, is_writer=True
        )
        rpc_broadcast_address = self.rpc_broadcast_mq.address

        # 3. 拉起所有 worker 进程
        unready: list[UnreadyWorkerProcHandle] = []
        for local_rank in range(self.local_world_size):
            rank = local_rank  # 单机多卡：local_rank == rank
            unready.append(
                WorkerProc.make_worker_process(
                    vllm_config=self.vllm_config,
                    local_rank=local_rank,
                    rank=rank,
                    distributed_init_method=distributed_init_method,
                    rpc_broadcast_address=rpc_broadcast_address,
                )
            )

        # 4. 等所有 worker 就绪（收集回复队列地址，按 rank 顺序排列）
        self.workers = WorkerProc.wait_for_ready(unready)
        self.response_mqs = [h.worker_response_mq for h in self.workers]

        # 5. 握手（顺序必须与 WorkerProc 保持一致，否则死锁）
        self.rpc_broadcast_mq.wait_until_ready()  # writer：收订阅 → 发 READY
        for response_mq in self.response_mqs:
            response_mq.wait_until_ready()  # reader：收 READY

        logger.info("执行器启动完成：%d 个 worker 已就绪", self.world_size)

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
        non_block: bool = False,
    ):
        """广播 method 给所有 worker，可同步收集或返回 Future。

        对应 vLLM 的 collective_rpc()。``non_block=True`` 时仅提交命令，
        返回值由有序 response executor 在后台核销。

        Args:
            method:           要调用的 worker 方法名（字符串）
            args / kwargs:    方法参数
            unique_reply_rank: 只收指定 rank 的回复；None 表示收所有 worker 的回复

        Returns:
            unique_reply_rank 给定时返回单个结果；否则返回 list（按 rank 顺序）。
        """
        assert self.rpc_broadcast_mq is not None, "执行器尚未初始化广播队列"

        kwargs = kwargs or {}

        # 1. 广播 RPC 请求
        self.rpc_broadcast_mq.enqueue((method, args, kwargs, unique_reply_rank))

        if non_block:
            return self._response_executor.submit(
                self._collect_rpc_responses,
                method,
                timeout,
                unique_reply_rank,
            )
        return self._collect_rpc_responses(method, timeout, unique_reply_rank)

    def _collect_rpc_responses(
        self,
        method: str,
        timeout: float | None,
        unique_reply_rank: int | None,
    ):
        """按广播顺序收集一条 RPC 的响应；由唯一 reader 串行调用。"""

        response_mqs = self.response_mqs
        if unique_reply_rank is not None:
            response_mqs = [response_mqs[unique_reply_rank]]

        responses = []
        for mq in response_mqs:
            status, result = mq.dequeue(timeout=timeout)
            if status != WorkerProc.ResponseStatus.SUCCESS:
                raise RuntimeError(f"worker 执行 {method} 失败：{result}")
            responses.append(result)

        return responses[0] if unique_reply_rank is not None else responses

    # ---- 常用 RPC 封装 ----

    def execute_model(self, scheduler_output, non_block: bool = False):
        """异步提交 forward；结果仅表示 execute RPC 是否成功。"""
        # 只有最后一个 PP stage 的第一个 TP rank 返回输出（预留 TP/PP）
        output_rank = self.world_size - self.parallel_config.tensor_parallel_size
        return self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            unique_reply_rank=output_rank,
            non_block=non_block,
        )

    def sample_tokens(self, non_block: bool = False):
        """异步提交独立采样 RPC，只回收最后 PP stage 的 driver 输出。"""

        output_rank = self.world_size - self.parallel_config.tensor_parallel_size
        return self.collective_rpc(
            "sample_tokens",
            unique_reply_rank=output_rank,
            non_block=non_block,
        )

    def get_kv_cache_specs(self) -> list:
        """收集每个 Worker 根据本 rank 模型得到的逐层 KV 规格。"""
        return self.collective_rpc("get_kv_cache_spec")

    def get_model_memory_usage(self) -> list[int]:
        """收集每个 Worker 的局部模型权重显存占用。"""
        return self.collective_rpc("get_model_memory_usage")

    def determine_available_memory(self) -> list[int]:
        """让所有 Worker 做 profiling，返回各 rank 的 KV 可用字节数。"""
        return self.collective_rpc("determine_available_memory")

    def initialize_from_config(self, kv_cache_configs) -> list:
        """把逐 rank KVCacheConfig 广播出去，由 Worker 选择自己的配置。"""
        if len(kv_cache_configs) != self.world_size:
            raise ValueError("KV cache config 数量必须等于 world_size")
        return self.collective_rpc(
            "initialize_from_config", args=(kv_cache_configs,)
        )

    def check_health(self) -> list:
        """健康检查（用于验证 RPC 通路）"""
        return self.collective_rpc("check_health", timeout=10)

    def shutdown(self) -> None:
        """关闭执行器：通知 worker 退出并清理消息队列"""
        # 1. 关闭 death_writer，通知各 worker 父进程已退出（子进程据此优雅关闭）
        for h in self.workers:
            if h.death_writer is not None:
                h.death_writer.close()
                h.death_writer = None

        # 2. 等待 worker 优雅退出，超时则 SIGTERM → SIGKILL
        self._ensure_worker_termination([h.proc for h in self.workers])

        # 3. 清理消息队列
        for h in self.workers:
            if h.worker_response_mq is not None:
                h.worker_response_mq.shutdown()
                h.worker_response_mq = None
        if self.rpc_broadcast_mq is not None:
            self.rpc_broadcast_mq.shutdown()
            self.rpc_broadcast_mq = None
        self._response_executor.shutdown(wait=False, cancel_futures=True)

        logger.info("执行器已关闭")

    def _ensure_worker_termination(self, procs: list[Process], grace_timeout: float = 3.0) -> None:
        """确保所有 worker 进程退出：先给时间优雅退出，超时 SIGTERM，再 SIGKILL"""
        start = time.time()
        while time.time() - start < grace_timeout:
            if all(not p.is_alive() for p in procs):
                return
            time.sleep(0.1)

        for p in procs:
            if p.is_alive():
                p.terminate()
        time.sleep(1)
        for p in procs:
            if p.is_alive():
                p.kill()
