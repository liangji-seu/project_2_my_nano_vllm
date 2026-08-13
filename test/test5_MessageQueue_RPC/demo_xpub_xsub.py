"""
单机多进程 demo: ZMQ XPUB / XSUB

演示 XPUB 相比普通 PUB 的核心能力:

  普通 PUB: 纯单向, 发布方不知道有没有人订阅 → slow-joiner 问题
  XPUB:     发布方能收到"订阅消息", 从而精确知道有几个读者在线

vLLM 正是用这一点, 在真正广播数据之前先 recv() 等所有 reader 的订阅消息,
从而彻底解决 slow-joiner, 不需要 time.sleep(0.2) 兜底。

订阅消息的字节格式 (XPUB 收到的是):
  b'\\x01<topic>'  订阅 topic   (\\x01 = subscribe)
  b'\\x00<topic>'  退订 topic   (\\x00 = unsubscribe)
  b'\\x01'         订阅所有     (topic 为空)

运行: python demo_xpub_xsub.py
依赖: pip install pyzmq
"""

import multiprocessing
import time

import zmq


def reader(addr):
    """子进程 (模拟 Worker): 用 XSUB 手动发订阅消息"""
    ctx = zmq.Context()
    # XSUB = 扩展版 SUB: 不用 setsockopt, 而是手动 send 订阅消息
    xsub = ctx.socket(zmq.XSUB)
    xsub.connect(addr)

    # 手动发订阅消息: \\x01 表示"订阅", 后面跟 topic(空=所有)
    xsub.send(b"\x01")
    print("[reader] 已连接, 手动发送订阅消息 b'\\x01' (订阅所有)")

    while True:
        topic, body = xsub.recv_multipart()  # 阻塞等数据
        if topic == b"stop":
            print("[reader] 收到 stop, 退出")
            break
        print(f"[reader] 收到: topic={topic.decode()}, body={body.decode()}")

    xsub.close()
    ctx.term()
    print("[reader] 退出")


def writer_loop(xpub, expected_readers):
    """父进程 (模拟 Executor): 等够订阅者, 再广播"""
    # 关键: 在 XPUB 上 recv 订阅消息, 数够 expected_readers 个才发数据
    seen = 0
    while seen < expected_readers:
        sub_msg = xpub.recv()  # 这里收到的是 b'\\x01' 之类的订阅消息
        seen += 1
        print(f"[writer] 收到订阅 #{seen}: {sub_msg!r}")

    print("[writer] 所有 reader 已订阅, 开始广播 (全程无 sleep 兜底)\n")

    for i in range(3):
        xpub.send_multipart([b"task", f"任务#{i}".encode()])
        print(f"[writer] 广播: 任务#{i}")
        time.sleep(0.1)

    xpub.send_multipart([b"stop", b""])
    print("[writer] 广播 stop")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    addr = "tcp://127.0.0.1:15556"

    # 父进程先建好 XPUB 并 bind, 再 spawn 子进程 —— 保证订阅消息不会丢
    ctx = zmq.Context()
    xpub = ctx.socket(zmq.XPUB)
    # 默认 XPUB 只在"第一个"订阅者来时报一次; 设 True 后"每个"都报,
    # 这样才能精确数到所有 reader。
    xpub.setsockopt(zmq.XPUB_VERBOSE, True)
    xpub.bind(addr)
    print(f"[writer] XPUB 绑定: {addr}")

    proc = multiprocessing.Process(target=reader, args=(addr,))
    proc.start()

    writer_loop(xpub, expected_readers=1)

    proc.join(timeout=3)
    if proc.is_alive():
        proc.terminate()
    xpub.close()
    ctx.term()
    print("[writer] 退出")
