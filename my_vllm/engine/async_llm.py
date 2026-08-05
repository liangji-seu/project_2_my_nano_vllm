"""
AsyncLLM — 异步引擎前端（当前为占位实现）
后续将对接真正的推理引擎
"""

import asyncio
import logging

from my_vllm.config import EngineConfig

logger = logging.getLogger(__name__)


class AsyncLLM:
    """
    异步 LLM 引擎前端

    对应 vLLM 的 AsyncLLM 类，负责：
    - 持有引擎配置
    - 提供模型信息
    - 转发推理请求到后端引擎（待实现）
    """

    def __init__(self, vllm_config: EngineConfig):
        self.vllm_config = vllm_config
        self.model_config = vllm_config
        self._errored = False
        logger.info(
            "AsyncLLM 初始化完成，模型: %s, 最大长度: %d",
            vllm_config.model,
            vllm_config.max_model_len,
        )

    @property
    def errored(self) -> bool:
        return self._errored

    async def get_supported_tasks(self) -> tuple[str, ...]:
        """返回支持的任务类型"""
        return ("generate",)

    async def do_log_stats(self) -> None:
        """模拟统计日志输出"""
        logger.info(
            "[Stats] 模型=%s, 状态=正常运行中",
            self.vllm_config.model,
        )

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> str:
        """
        占位推理 — 后续对接真实引擎

        当前返回一个模拟回复
        """
        logger.info("收到推理请求: prompt=%.50s..., max_tokens=%d", prompt, max_tokens)
        # 模拟推理延迟
        await asyncio.sleep(0.1)
        return f"[占位回复] 你输入的是: {prompt[:50]}..."

    async def shutdown(self, timeout: float | None = None) -> None:
        """关闭引擎"""
        logger.info("AsyncLLM 正在关闭...")
        await asyncio.sleep(0.1)
        logger.info("AsyncLLM 已关闭")
