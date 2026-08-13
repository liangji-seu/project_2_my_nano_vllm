"""
单机多进程 demo: 共享内存 (Shared Memory) + ZMQ 通知

模拟 vLLM Executor → Worker 的通信模式:
  1. 父进程 (Executor): 写数据到共享内存, 发 ZMQ 通知
  2. 子进程 (Worker):  收到 ZMQ 通知, 从共享内存读数据

运行: python demo_shm_zmq.py
依赖: pip install pyzmq
"""

import multiprocessing
import multiprocessing.shared_memory as shm
import struct
import time

import zmq

# ============================================================
# 共享内存区的简单布局
# ============================================================
# [0..3]    : 4 bytes — 数据长度 (大端 uint32)
# [4..end]  : 实际数据 (变长, 最大 4092 bytes)
# ============================================================
SHM_SIZE = 4096


def parent(recv_pipe, send_pipe):
    """
    父进程 (模拟 Executor)
    recv_pipe: 可读端, 收子进程的就绪信号
    send_pipe: 可写端, 传给子进程用于发就绪信号
    """
    # === 1. 创建共享内存 ===
    shm_buf = shm.SharedMemory(create=True, size=SHM_SIZE)
    print(f"[parent] 创建共享内存: name={shm_buf.name}, size={SHM_SIZE}")

    # === 2. 创建 ZMQ PUB socket (通知子进程) ===
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind("tcp://127.0.0.1:15555")
    print("[parent] ZMQ PUB 绑定: tcp://127.0.0.1:15555")

    # === 3. 把共享内存名字和 ZMQ 地址传给子进程 ===
    #      spawn 方式下, 通过函数参数传递这些"连接信息"
    #      类比 vLLM 的 Handle
    handle = {
        "shm_name": shm_buf.name,
        "zmq_addr": "tcp://127.0.0.1:15555",
    }

    # === 4. 启动子进程 ===
    proc = multiprocessing.Process(target=child, args=(send_pipe, handle))
    proc.start()

    # 等待子进程就绪
    recv_pipe.recv()
    # ZMQ PUB/SUB slow-joiner: SUB 订阅需要一个小窗口, 稍微等一下
    time.sleep(0.2)
    print("[parent] 子进程已就绪\n")

    # === 5. 写数据 → 通知 → 写数据 → 通知 (多轮) ===
    for i in range(3):
        msg = f"调度任务 #{i}: 请处理这批次请求".encode("utf-8")
        data_len = len(msg)

        # 写共享内存: [长度(4B)][数据]
        shm_buf.buf[0:4] = struct.pack(">I", data_len)
        shm_buf.buf[4 : 4 + data_len] = msg
        print(f"[parent] 写入共享内存: {msg.decode()}")

        # ZMQ: 话题 "task" + 消息体 "新任务来了"
        pub.send_multipart([b"task", "新任务来了".encode()])

        time.sleep(0.5)

    # === 6. 发送停止信号 ===
    shm_buf.buf[0:4] = struct.pack(">I", 0)
    pub.send_multipart([b"stop", "结束".encode()])
    print("[parent] 发送停止信号")

    proc.join(timeout=3)
    if proc.is_alive():
        proc.terminate()
    shm_buf.close()
    shm_buf.unlink()
    pub.close()
    ctx.term()
    print("[parent] 退出")


def child(ready_pipe, handle):
    """
    子进程 (模拟 Worker)
    """
    # === 1. 连接共享内存 ===
    shm_buf = shm.SharedMemory(name=handle["shm_name"])
    print(f"[child] 连接共享内存: name={shm_buf.name}")

    # === 2. 连接 ZMQ SUB socket ===
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(handle["zmq_addr"])
    sub.setsockopt_string(zmq.SUBSCRIBE, "task")  # 订阅 task 话题
    sub.setsockopt_string(zmq.SUBSCRIBE, "stop")  # 订阅 stop 话题
    print(f"[child] ZMQ SUB 连接: {handle['zmq_addr']}")

    # 通知父进程已就绪
    ready_pipe.send("ready")

    # === 3. 循环等待通知 → 读共享内存 ===
    while True:
        topic, body = sub.recv_multipart()  # 阻塞等待, 拿到 [话题, 消息体]
        print(f"[child] ZMQ 收到: topic={topic.decode()}, body={body.decode()}")

        if topic == b"stop":
            print("[child] 收到停止信号, 退出")
            break

        # 是 "task" 话题, 读共享内存
        data_len = struct.unpack(">I", shm_buf.buf[0:4].tobytes())[0]
        msg = shm_buf.buf[4 : 4 + data_len].tobytes().decode("utf-8")
        print(f"[child] 从共享内存读到任务: {msg}")

    shm_buf.close()
    sub.close()
    ctx.term()
    print("[child] 退出")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # Pipe(duplex=False): 返回 (可读端, 可写端)
    recv_pipe, send_pipe = multiprocessing.Pipe(duplex=False)

    # 父进程持可读端等通知, 可写端传给子进程
    parent(recv_pipe, send_pipe)
    recv_pipe.close()
