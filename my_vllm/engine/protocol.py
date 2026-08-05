"""
EngineClient 协议
定义引擎前端的接口契约 — 后续真实引擎需要实现此协议
"""

from typing import Any, Protocol, runtime_checkable

from my_vllm.config import EngineConfig


@runtime_checkable
class EngineClient(Protocol):
    """引擎客户端协议，API 层通过此接口与引擎通信"""

    @property
    def vllm_config(self) -> EngineConfig: ...

    @property
    def model_config(self) -> Any: ...

    @property
    def errored(self) -> bool: ...

    async def get_supported_tasks(self) -> tuple[str, ...]:
        """返回引擎支持的任务类型列表，如 ('generate',)"""
        ...

    async def do_log_stats(self) -> None:
        """触发一次统计日志输出"""
        ...

    async def shutdown(self, timeout: float | None = None) -> None:
        """关闭引擎，释放资源"""
        ...
