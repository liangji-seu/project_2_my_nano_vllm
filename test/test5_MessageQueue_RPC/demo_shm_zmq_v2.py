"""
单机多进程 demo v2: 共享内存 + ZMQ 通知 + 序列号同步

模拟 vLLM Executor → Worker 的通信模式:
  1. 父进程 (Executor): 写数据到共享内存, 发 ZMQ 通知, 等子进程 ACK
  2. 子进程 (Worker):  收到 ZMQ 通知, 读共享内存, 发 ACK 确认读完

v2 改进: 解决 v1 中"子进程不知道读的是不是父进程刚写的那次"的竞争问题

同步机制:
  - 共享内存里加了序列号 (每轮递增), 子进程可以验证
  - 父进程写完后等子进程 ACK, 才写下一轮 (避免覆盖未读数据)

运行: python demo_shm_zmq.py
依赖: pip install pyzmq
"""

import multiprocessing
import multiprocessing.shared_memory as shm
import struct
import time
from multiprocessing import get_context

import zmq

# ============================================================
# 共享内存布局 (总大小 4096 bytes)
# ============================================================
# [0..3]    : 4 bytes — 序列号 round_id (大端 uint32)
# [4..7]    : 4 bytes — 数据长度 (大端 uint32)
# [8..end]  : 实际数据 (变长, 最大 4088 bytes)
# ============================================================
SHM_SIZE = 4096
HEADER_LEN = 4              # 序列号
LENGTH_LEN = 4              # 数据长度
DATA_OFFSET = HEADER_LEN + LENGTH_LEN  # 数据起始偏移 = 8


def parent(recv_pipe, child_pipe_for_send):
    """
    父进程 (模拟 Executor)

    recv_pipe:         可读端, 收子进程的 ready 和 ack 信号
    child_pipe_for_send: 可写端, 传给子进程用于发信号
    """
    # === 1. 创建共享内存 ===
    shm_buf = shm.SharedMemory(create=True, size=SHM_SIZE)
    # 初始化序列号为 0
    shm_buf.buf[0:HEADER_LEN] = struct.pack(">I", 0)
    print(f"[parent] 创建共享内存: name={shm_buf.name}, size={SHM_SIZE}")

    # === 2. 创建 ZMQ PUB socket ===
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind("tcp://127.0.0.1:15555")
    print("[parent] ZMQ PUB 绑定: tcp://127.0.0.1:15555")

    # === 3. 传给子进程的"连接说明书" (类比 vLLM Handle) ===
    handle = {
        "shm_name": shm_buf.name,
        "zmq_addr": "tcp://127.0.0.1:15555",
    }

    # === 4. 启动子进程 ===
    ctx_mp = get_context("spawn")
    proc = ctx_mp.Process(target=child, args=(child_pipe_for_send, handle))
    proc.start()

    # 等待子进程就绪: 子进程连上 shm 和 ZMQ 后发 "ready"
    recv_pipe.recv()
    time.sleep(0.2)  # ZMQ PUB/SUB slow-joiner 兜底
    print("[parent] 子进程已就绪\n")

    # === 5. 写数据 → ZMQ通知 → 等ACK → 下一轮 ===
    for i in range(3):
        msg = f"调度任务 #{i}: 请处理这批次请求".encode("utf-8")
        data_len = len(msg)

        # 写共享内存: [序列号][长度][数据]
        shm_buf.buf[HEADER_LEN:HEADER_LEN + LENGTH_LEN] = struct.pack(">I", data_len)
        shm_buf.buf[DATA_OFFSET:DATA_OFFSET + data_len] = msg
        # 序列号最后写 — 子进程读到新序列号就知道数据就绪
        new_round = i + 1
        shm_buf.buf[0:HEADER_LEN] = struct.pack(">I", new_round)

        print(f"[parent] 写入 shm (round={new_round}): {msg.decode()}")

        # ZMQ 通知: "有新数据了"
        pub.send(b"new_data")

        # 等子进程 ACK: 确认它已经读完了, 父进程才能安全覆盖下一轮数据
        ack = recv_pipe.recv()
        print(f"[parent] 收到 ACK, round={ack}")

    # === 6. 发送停止信号 ===
    shm_buf.buf[HEADER_LEN:HEADER_LEN + LENGTH_LEN] = struct.pack(">I", 0)
    shm_buf.buf[0:HEADER_LEN] = struct.pack(">I", 0xFFFFFFFF)
    pub.send(b"new_data")
    print("[parent] 发送停止信号")

    proc.join(timeout=5)
    if proc.is_alive():
        proc.terminate()
    shm_buf.close()
    shm_buf.unlink()
    pub.close()
    ctx.term()
    print("[parent] 退出")


def child(write_pipe, handle):
    """
    子进程 (模拟 Worker)

    write_pipe: 可写端, 用于向父进程发信号 (ready / ack)
    handle:     包含 shm_name 和 zmq_addr 的连接信息
    """
    last_round = 0  # 追踪上轮读到的序列号

    # === 1. 连接共享内存 ===
    shm_buf = shm.SharedMemory(name=handle["shm_name"])
    print(f"[child] 连接共享内存: name={shm_buf.name}")

    # === 2. 连接 ZMQ SUB socket ===
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(handle["zmq_addr"])
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    print(f"[child] ZMQ SUB 连接: {handle['zmq_addr']}")

    # 通知父进程: 我已就绪
    write_pipe.send("ready")

    # === 3. 循环: 等 ZMQ 通知 → 读 shm → 发 ACK ===
    while True:
        _ = sub.recv()  # 阻塞等通知

        # 读序列号, 确认是新的一轮
        round_id = struct.unpack(">I", shm_buf.buf[0:HEADER_LEN].tobytes())[0]

        # 停止信号
        if round_id == 0xFFFFFFFF:
            print("[child] 收到停止信号, 退出")
            break

        if round_id == last_round:
            # ZMQ 通知是旧的 (可能积压), 跳过
            continue

        last_round = round_id

        # 读数据长度和内容
        data_len = struct.unpack(
            ">I", shm_buf.buf[HEADER_LEN:HEADER_LEN + LENGTH_LEN].tobytes()
        )[0]
        msg = shm_buf.buf[DATA_OFFSET:DATA_OFFSET + data_len].tobytes().decode("utf-8")
        print(f"[child] 收到任务 (round={round_id}): {msg}")

        # 发 ACK: 告诉父进程"我已读完, 你可以写下一轮了"
        write_pipe.send(round_id)

    shm_buf.close()
    sub.close()
    ctx.term()
    print("[child] 退出")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    recv_pipe, send_pipe = multiprocessing.Pipe(duplex=False)
    parent(recv_pipe, send_pipe)
    recv_pipe.close()
