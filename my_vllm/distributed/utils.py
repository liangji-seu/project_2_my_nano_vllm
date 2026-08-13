"""分布式相关的小工具函数"""

import socket
import uuid


def get_open_port() -> int:
    """找一个空闲的 TCP 端口（用于 NCCL 的 rendezvous 握手地址）"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_open_zmq_ipc_path() -> str:
    """生成一个唯一的 zmq IPC 地址（本机 Unix domain socket 路径）"""
    return f"ipc:///tmp/my_vllm_{uuid.uuid4().hex}.ipc"
