"""
AsyncLLM — 异步引擎前端

对应 vLLM 的 vllm/v1/engine/async_llm.py

AsyncLLM 是 API 层和引擎后端之间的桥梁:
  - 持有引擎配置
  - 通过 MPClient 管理 ZMQ 通信 + 引擎进程
  - 提供 generate() 接口给 API 路由

完整请求流:
  HTTP POST → chat_completions()
      → AsyncLLM.generate()          ← 构造请求, 通过 ZMQ 发送
          → MPClient.generate()      ← ROUTER → DEALER
              → EngineCoreProc       ← 引擎进程处理
          ← MPClient._process_outputs()  ← PULL ← PUSH
      ← StreamingResponse (SSE)      ← 逐 token 返回给客户端
"""

import asyncio
import logging

from my_vllm.config import EngineConfig
from my_vllm.engine.core_client import MPClient

logger = logging.getLogger(__name__)


class AsyncLLM:
    """异步 LLM 引擎前端

    对应 vLLM 的 AsyncLLM

    生命周期:
      1. __init__()    构造 MPClient（创建 ZMQ sockets + 启动引擎进程）
      2. start()       握手等待引擎 READY + 启动输出处理任务
      3. generate()    处理推理请求
      4. shutdown()    清理

    后续 commit 增加:
      - self.renderer:       OnlineRenderer (tokenizer + chat template)
      - self.input_processor: InputProcessor (EngineInput → EngineCoreRequest)
      - self.output_processor: OutputProcessor (EngineCoreOutput → RequestOutput)
    """

    def __init__(self, vllm_config: EngineConfig):
        self.vllm_config = vllm_config
        self.model_config = vllm_config
        self._errored = False
        self._started = False

        engine_count = vllm_config.data_parallel_size
        logger.info(
            "AsyncLLM 正在构造 (model=%s, engine_count=%d)...",
            vllm_config.model,
            engine_count,
        )

        # 构造传输层 — 创建 ZMQ sockets + 启动引擎进程
        # 对应 vLLM: self.engine_core = AsyncMPClient(...)
        self.mp_client = MPClient(
            vllm_config=vllm_config,
            engine_count=engine_count,
        )

        logger.info(
            "AsyncLLM 构造完成 (model=%s, engine_count=%d) — "
            "请在 event loop 中调用 start() 完成握手",
            vllm_config.model,
            engine_count,
        )

    @property
    def errored(self) -> bool:
        return self._errored or self.mp_client.engine_dead

    # ---- 启动 ----

    async def start(self) -> None:
        """启动引擎前端 — 握手 + 输出处理

        必须在 asyncio event loop 中调用（即 API server 启动后）
        """
        if self._started:
            return

        # 握手等待引擎 READY + 启动输出处理任务
        await self.mp_client.start()
        self._started = True
        logger.info("AsyncLLM 启动完成")

    # ---- 模型信息 ----

    async def get_supported_tasks(self) -> tuple[str, ...]:
        """返回引擎支持的任务类型"""
        return ("generate",)

    async def do_log_stats(self) -> None:
        """输出统计日志"""
        alive = sum(
            1
            for p in self.mp_client._engine_manager.processes
            if p.is_alive()
        )
        logger.info(
            "[Stats] 模型=%s, 引擎进程=%d/%d 存活",
            self.vllm_config.model,
            alive,
            len(self.mp_client._engine_manager.processes),
        )

    # ---- 推理接口 ----

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> str:
        """通过 ZMQ 发送推理请求到引擎后端, 等待返回结果

        对应 vLLM AsyncLLM.generate()

        流程:
          1. 确保已启动（懒 start, 防止忘记调用 start()）
          2. 通过 mp_client 发送 ROUTER → DEALER
          3. 等待 PULL ← PUSH 返回结果

        Args:
            prompt:     文本 prompt
            max_tokens: 最大生成 token 数（预留）
            temperature: 采样温度（预留）

        Returns:
            引擎生成的文本
        """
        # 懒启动 — 如果还没调用 start(), 自动启动
        if not self._started:
            await self.start()

        logger.info(
            "收到推理请求: prompt=%.50s..., max_tokens=%d",
            prompt,
            max_tokens,
        )

        # 通过 ZMQ 发送到引擎后端, 等待返回
        result = await self.mp_client.generate(prompt)
        logger.info("推理完成: result=%.50s...", result)
        return result

    # ---- 关闭 ----

    async def shutdown(self, timeout: float | None = None) -> None:
        """关闭引擎前端, 清理所有资源"""
        logger.info("AsyncLLM 正在关闭...")
        self._started = False

        # 关闭传输层（终止引擎进程 + 清理 ZMQ）
        self.mp_client.shutdown()
        logger.info("AsyncLLM 已关闭")
