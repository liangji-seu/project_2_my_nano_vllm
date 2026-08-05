"""
配置系统：EngineConfig + CLI 参数解析
"""

import argparse
from dataclasses import dataclass, field


@dataclass
class EngineConfig:
    """引擎配置 — 从 CLI 参数构造"""

    host: str = "0.0.0.0"
    port: int = 8000
    model: str = "test-model"
    disable_log_stats: bool = False
    served_model_name: str | None = None
    max_model_len: int = 4096

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "EngineConfig":
        return cls(
            host=args.host,
            port=args.port,
            model=args.model,
            disable_log_stats=args.disable_log_stats,
            served_model_name=args.served_model_name,
            max_model_len=args.max_model_len,
        )


def make_arg_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数解析器"""
    parser = argparse.ArgumentParser(
        description="my_vllm — 极简在线推理服务"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="服务器监听地址"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="服务器监听端口"
    )
    parser.add_argument(
        "--model", type=str, default="test-model", help="模型名称或路径"
    )
    parser.add_argument(
        "--disable-log-stats",
        action="store_true",
        help="禁用定期统计日志",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="对外暴露的模型名称",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="模型最大上下文长度",
    )
    parser.add_argument(
        "--enable-log-requests",
        action="store_true",
        help="启用请求日志",
    )
    return parser
