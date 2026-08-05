"""
c_s_2.py  —  多 DEALER 线程 + 单 ROUTER 线程的"随意通信"演示

对比 c_s.py，本文件刻意展示 ROUTER/DEALER 异步模式的四个关键特性：
    ① DEALER 主动发消息，不等回复（不是一问一答）
    ② ROUTER 主动向任意 DEALER 推送消息，不需要 DEALER 先请求
    ③ ROUTER 作为中转，把一个 DEALER 的消息转发给另一个 DEALER
       （DEALER 之间互相不知道对方的存在，只有 ROUTER 掌握全部 identity）
    ④ ROUTER 的 identity 学习机制：收到某 DEALER 的第一帧消息后才认识它，
       之后才能向它定向发送（这就是 vLLM 中引擎上线要发 ready 的原因）

结构：
    ROUTER 线程（1 个）：poller 多路监听 + 定时主动广播
    DEALER 线程（N 个）：每线程一个 DEALER，各带唯一 identity
                         每路再挂一个收线程，展示"收到的消息不分回复/推送/转发"
"""

import threading
import time
import uuid

import zmq

PORT = 6667


# ============================== ROUTER 端（单线程） ==============================
def router():
    context = zmq.Context.instance()
    frontend = context.socket(zmq.ROUTER)  # 1 个 ROUTER 对 N 个 DEALER
    frontend.bind(f"tcp://localhost:{PORT}")
    print(f"[ROUTER] 启动，bind {PORT}")

    known: dict[bytes, float] = {}  # identity -> 上线时间（ROUTER 的路由表）
    lock = threading.Lock()

    poller = zmq.Poller()
    poller.register(frontend, zmq.POLLIN)

    last_broadcast = time.monotonic()
    try:
        while True:
            socks = dict(poller.poll(200))  # 200ms 超时，顺带当定时器
            now = time.monotonic()

            if frontend in socks:
                identity, *rest = frontend.recv_multipart()
                msg = zmq.utils.jsonapi.loads(rest[-1]) if rest else {}
                mtype = msg.get("type")

                # ④ identity 学习：DEALER 的第一帧消息让 ROUTER 记住它
                with lock:
                    if identity not in known:
                        known[identity] = now
                        print(f"[ROUTER] ①学习到新 DEALER: {identity.decode()} "
                              f"（路由表现在共 {len(known)} 个）")

                if mtype == "hello":
                    # ROUTER 也可以回复，但这不是协议义务 —— 展示"想回就回"
                    frontend.send_multipart([
                        identity,
                        zmq.utils.jsonapi.dumps({
                            "type": "reply",
                            "content": f"你好 {identity.decode()}，ROUTER 已认识你",
                        }),
                    ])
                elif mtype == "relay":
                    # ③ 中转转发：A 发来的消息，ROUTER 定向发给 B
                    target = msg["target"].encode()
                    with lock:
                        if target in known:
                            frontend.send_multipart([
                                target,
                                zmq.utils.jsonapi.dumps({
                                    "type": "forwarded",
                                    "from": identity.decode(),
                                    "content": msg["content"],
                                }),
                            ])
                            print(f"[ROUTER] ③中转: {identity.decode()} -> {msg['target']}")
                        else:
                            # ROUTER 不能给不认识的 identity 发消息（会被静默丢弃）
                            print(f"[ROUTER] ③中转失败: 目标 {msg['target']} 还没上线（未学习到）")
                else:
                    print(f"[ROUTER] 收到 {identity.decode()} 的普通消息: "
                          f"{msg.get('content')}（不回也不影响继续通信）")

            # ② ROUTER 主动推送：无需任何 DEALER 请求，定时广播
            if now - last_broadcast >= 2.0:
                last_broadcast = now
                with lock:
                    targets = list(known.keys())
                for t in targets:
                    frontend.send_multipart([
                        t,
                        zmq.utils.jsonapi.dumps({
                            "type": "broadcast",
                            "content": f"ROUTER 主动推送 @{time.strftime('%H:%M:%S')}",
                        }),
                    ])
                if targets:
                    print(f"[ROUTER] ②主动广播给 {len(targets)} 个 DEALER（无需它们请求）")
    except KeyboardInterrupt:
        print("[ROUTER] 退出")
    finally:
        frontend.close()


# ============================== DEALER 端（每线程一个） ==============================
def dealer(worker_id: int):
    context = zmq.Context.instance()
    sock = context.socket(zmq.DEALER)
    identity = f"worker-{worker_id}".encode()
    sock.setsockopt(zmq.IDENTITY, identity)
    sock.connect(f"tcp://localhost:{PORT}")

    # 收消息的线程：DEALER 收到的消息不分"回复/推送/转发"，统统是异步到达的消息
    def recv_loop():
        while True:
            msg = zmq.utils.jsonapi.loads(sock.recv())
            print(f"    [{identity.decode()}] 收到[{msg['type']}] {msg['content']}")

    threading.Thread(target=recv_loop, daemon=True).start()

    # ① 上线：DEALER 主动发 hello，让 ROUTER 学习 identity（对应 vLLM 的 ready 消息）
    sock.send_json({"type": "hello", "content": "我上线了"})

    # ① 异步连发 3 条，不等任何回复 —— DEALER 不是一问一答
    for i in range(3):
        sock.send_json({
            "type": "query",
            "msg_id": str(uuid.uuid4()),
            "content": f"请求{i} 来自 {identity.decode()}",
        })
    print(f"[{identity.decode()}] ①连发 3 条查询后立刻返回（不等待回复）")

    # ③ 让 worker-1 晚一点发一条转发消息给 worker-2（两者互不知道对方）
    if worker_id == 1:
        time.sleep(1.0)
        sock.send_json({
            "type": "relay",
            "target": "worker-2",
            "content": "你好 worker-2，我是 worker-1，借 ROUTER 的通道向你问好",
        })


if __name__ == "__main__":
    t_start = time.time()

    router_thread = threading.Thread(target=router, daemon=True)
    router_thread.start()
    time.sleep(0.5)

    dealer_threads = []
    for i in range(3):
        t = threading.Thread(target=dealer, args=(i,), daemon=True)
        t.start()
        dealer_threads.append(t)

    print(f"[{time.time() - t_start:.3f}s] 1 个 ROUTER 线程 + 3 个 DEALER 线程已启动，观察 8 秒\n")
    time.sleep(8)

    print(f"\n[{time.time() - t_start:.3f}s] 演示结束")
    print("=" * 60)
    print("要点回顾：")
    print("  ① DEALER 主动连发消息，不等回复 → 异步")
    print("  ② ROUTER 定时主动广播 → 没有'先请求后响应'的协议约束")
    print("  ③ worker-1 的消息经 ROUTER 中转给 worker-2 → 定向路由靠 identity")
    print("  ④ ROUTER 先学习 identity 才能定向发送 → vLLM 引擎上线发 ready 的原因")
    print("=" * 60)
