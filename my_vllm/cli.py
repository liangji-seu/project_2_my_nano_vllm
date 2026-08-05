"""
CLI 入口：my_vllm serve

用法:
    my_vllm serve --model test-model --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import sys

from my_vllm.config import make_arg_parser
from my_vllm.entrypoints.api_server import run_server

logger = logging.getLogger(__name__)


def main():
    """my_vllm CLI 主入口"""

    parser = make_arg_parser()

    # 确保至少有一个子命令（当前只支持 serve）
    if len(sys.argv) == 1 or sys.argv[1] not in ("serve", "-h", "--help"):
        print("用法: my_vllm serve [选项]")
        print("       my_vllm serve --model <模型名> --host <地址> --port <端口>")
        print()
        print("示例: my_vllm serve --model test-model --host 0.0.0.0 --port 8000")
        sys.exit(1)

    if sys.argv[1] in ("-h", "--help"):
        parser.print_help()
        sys.exit(0)

    # my_vllm serve → 跳过 "serve" 子命令名，解析剩余参数
    args = parser.parse_args(sys.argv[2:])

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("启动 my_vllm serve, 模型=%s, 端口=%d", args.model, args.port)

    # 启动事件循环（用标准 asyncio，不用 uvloop）
    asyncio.run(run_server(args))
