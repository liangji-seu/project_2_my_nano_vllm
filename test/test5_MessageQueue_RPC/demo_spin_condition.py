"""
单机多进程 demo: 用普通 PUB/SUB 实现 SpinCondition 的 notify socket

模拟 vLLM 的 SpinCondition:
  写者写完共享内存后, 在 notify socket (PUB) 上发一个字节 b'\\x00' 当"闹钟"
  读者收到后从共享内存读数据

核心知识点:
  1. notify socket 就是普通 PUB/SUB, 只传 1 字节信号, 不传数据
  2. 写端 PUB 设 SNDHWM=1    (忙时 ping 可以丢, 丢了没关系)
  3. 读端 SUB 设 CONFLATE=1  (只保留最新一条通知, 不堆积)
  4. "Spin": 读者先忙等 1 秒 (直接查共享内存, 不碰 socket),
     超时才进入 idle, 阻塞在 poller 上等 notify
  5. cancel: 本进程内 PAIR socket 用来唤醒阻塞的读者, 实现干净退出

运行: python demo_spin_condition.py
依赖: pip install pyzmq
"""

import multiprocessing
import multiprocessing.shared_memory as shm
import os
import struct
import threading
import time

import zmq

SHM_SIZE = 4096


def writer(shm_name, notify_addr, recv_pipe):
    """父进程 (模拟 Executor): 写共享内存 + 发 notify ping"""
    shm_buf = shm.SharedMemory(name=shm_name)
    ctx = zmq.Context()

    # notify socket: 普通 PUB
    notify = ctx.socket(zmq.PUB)
    # 发送高水位=1: 忙时后续 ping 静默丢弃也没关系 (通知丢了无所谓, 数据在共享内存)
    notify.setsockopt(zmq.SNDHWM, 1)
    notify.bind(notify_addr)

    recv_pipe.recv()  # 等 reader 就绪
    time.sleep(0.2)   # 普通 PUB 仍需 slow-joiner 兜底 (这正是 XPUB 要解决的问题)
    print("[writer] 就绪\n")

    print("[writer] 快速写 5 条 (reader 处于 spin 阶段, 直接查共享内存读到, 不阻塞 socket)")
    for i in range(1, 6):
        shm_buf.buf[0:4] = struct.pack(">I", i)
        notify.send(b"\x00")  # ping: 只发 1 字节, 不传数据
        print(f"[writer] 写 count={i}, 发 ping")
        time.sleep(0.1)

    print("\n[writer] 停 2 秒, 让 reader 进入 idle (阻塞等通知)...")
    time.sleep(2)

    print("\n[writer] 再写 1 条 (reader 此刻在 idle, 靠 notify 唤醒)")
    shm_buf.buf[0:4] = struct.pack(">I", 99)
    notify.send(b"\x00")
    print("[writer] 写 count=99, 发 ping")

    # 等 reader 侧的 cancel 线程触发退出
    time.sleep(3)
    shm_buf.close()
    notify.close()
    ctx.term()
    print("[writer] 退出")


def reader(shm_name, notify_addr, cancel_addr, ready_pipe):
    """子进程 (模拟 Worker): spin 忙等 + idle 阻塞等 notify"""
    shm_buf = shm.SharedMemory(name=shm_name)
    ctx = zmq.Context()

    # notify socket: 普通 SUB
    notify = ctx.socket(zmq.SUB)
    # CONFLATE=1: 只保留最新一条通知, 高负载下不堆积
    notify.setsockopt(zmq.CONFLATE, 1)
    notify.setsockopt_string(zmq.SUBSCRIBE, "")  # 订阅所有
    notify.connect(notify_addr)

    # cancel socket: 本进程内 inproc PAIR, 用于唤醒阻塞的读者
    cancel_in = ctx.socket(zmq.PAIR)
    cancel_in.bind(cancel_addr)

    poller = zmq.Poller()
    poller.register(notify, zmq.POLLIN)      # 等 notify
    poller.register(cancel_in, zmq.POLLIN)   # 等 cancel

    busy_loop_s = 1.0
    last_read = time.monotonic()
    last_count = 0

    # 后台线程: 模拟 vLLM 里"同进程的监控线程"在关停时唤醒读者
    def canceller():
        time.sleep(6)
        print("\n[reader] (后台线程) 发 cancel 唤醒读者")
        c = ctx.socket(zmq.PAIR)
        c.connect(cancel_addr)
        c.send(b"\x00")

    threading.Thread(target=canceller, daemon=True).start()

    ready_pipe.send("ready")
    print("[reader] 就绪, 开始读循环 (busy_loop_s=1s)\n")

    while True:
        now = time.monotonic()

        if now - last_read <= busy_loop_s:
            # ---- SPIN 阶段: 忙等, 直接查共享内存, 不阻塞 socket ----
            os.sched_yield()  # 让出 CPU 但不睡觉, 保持高频轮询
            count = struct.unpack(">I", shm_buf.buf[0:4].tobytes())[0]
            if count != last_count:
                last_count = count
                last_read = time.monotonic()
                print(f"[reader] (spin 直接读到) count={count}")
        else:
            # ---- IDLE 阶段: 阻塞等 notify 或 cancel ----
            events = dict(poller.poll(timeout=2000))
            if cancel_in in events:
                print("[reader] 收到 cancel, 退出")
                break
            elif notify in events:
                # CONFLATE 保证只有一条通知, 读掉它
                notify.recv(flags=zmq.NOBLOCK, copy=False)
                count = struct.unpack(">I", shm_buf.buf[0:4].tobytes())[0]
                if count != last_count:
                    last_count = count
                    last_read = time.monotonic()
                    print(f"[reader] (idle 被 notify 唤醒读到) count={count}")
            else:
                print("[reader] (idle poll 超时, 继续等)")

    shm_buf.close()
    notify.close()
    cancel_in.close()
    ctx.term()
    print("[reader] 退出")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    shm_buf = shm.SharedMemory(create=True, size=SHM_SIZE)
    shm_buf.buf[0:4] = struct.pack(">I", 0)
    notify_addr = "tcp://127.0.0.1:15557"
    cancel_addr = "inproc://cancel"

    recv_pipe, send_pipe = multiprocessing.Pipe(duplex=False)

    proc = multiprocessing.Process(
        target=reader, args=(shm_buf.name, notify_addr, cancel_addr, send_pipe)
    )
    proc.start()

    writer(shm_buf.name, notify_addr, recv_pipe)

    proc.join(timeout=5)
    if proc.is_alive():
        proc.terminate()
    shm_buf.close()
    shm_buf.unlink()
    recv_pipe.close()
    print("[main] 退出")
