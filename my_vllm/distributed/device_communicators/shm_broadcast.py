"""
简化版进程间广播消息队列 MessageQueue

对应 vLLM 的 vllm/distributed/device_communicators/shm_broadcast.py

简化说明（相比 vLLM 源码）：
  - vLLM 用「共享内存 ring buffer + zmq XPUB/XSUB 通知 + SpinCondition」实现
    零拷贝广播：大张量直接写共享内存，zmq 只传「哪个槽位可读」的通知。
  - 这里简化为「zmq XPUB/SUB + pickle 序列化」，数据直接走 socket。
    好处是简单易懂；代价是大张量会多一次序列化拷贝（学习项目可接受）。
  - 但保留了 wait_until_ready() 的「订阅同步握手」——这是 PUB/SUB 可靠性的关键。

通信模型：
  1 个 writer（XPUB，bind）→ N 个 reader（SUB，connect）

  writer.enqueue(obj)  → pickle 序列化后广播
  reader.dequeue()     → 接收并反序列化

为什么需要 wait_until_ready()（订阅同步握手）：
  zmq 的 PUB/SUB 有一个「慢连接」问题：SUB 端刚 connect/subscribe 时，
  订阅注册是异步的，此时 PUB 发的消息可能被静默丢弃。
  解决办法：writer 用 XPUB + XPUB_VERBOSE，能收到每个 reader 发来的订阅消息；
  等到收够 N 条订阅消息（= 所有 reader 都已连上并订阅），
  再广播 b"READY"，reader 收到 READY 即确认通道真正打通、不会丢消息。
"""

import pickle

import zmq

from my_vllm.distributed.utils import get_open_zmq_ipc_path


class MessageQueue:
    """一对多广播消息队列（1 writer → N reader）

    writer 端用 XPUB（PUB 的升级版，能接收订阅消息），reader 端用 SUB。
    """

    def __init__(
        self,
        n_reader: int,
        is_writer: bool,
        address: str | None = None,
    ):
        """
        Args:
            n_reader:   reader 数量。writer 端用它决定握手时要等多少条订阅消息。
            is_writer:  True 表示本进程是 writer（XPUB bind）；False 是 reader（SUB connect）。
            address:    zmq 地址。writer 不传则自动生成一个 IPC 路径并 bind；
                        reader 必须传入 writer 的地址来 connect。
        """
        self.n_reader = n_reader
        self.is_writer = is_writer
        self._context = zmq.Context()

        if is_writer:
            # XPUB = PUB 的升级版，能接收 reader 发来的订阅消息
            self._socket = self._context.socket(zmq.XPUB)
            # XPUB_VERBOSE：每个订阅/退订都上报，否则只上报第一个订阅
            self._socket.setsockopt(zmq.XPUB_VERBOSE, True)
            if address is None:
                address = get_open_zmq_ipc_path()
            self._socket.bind(address)
            self.address = address
        else:
            self._socket = self._context.socket(zmq.SUB)
            self._socket.setsockopt_string(zmq.SUBSCRIBE, "")  # 订阅所有消息
            assert address is not None, "reader 端必须提供 address"
            self._socket.connect(address)
            self.address = address

    def wait_until_ready(self) -> None:
        """订阅同步握手（writer 和 reader 都要调用，且顺序必须一致）

        writer 端：
          1. 阻塞接收 N 条订阅消息（每条 = reader 发来的 1 字节 b'\\x01'）
             —— 收够 N 条 = 所有 reader 都已连上并订阅
          2. 广播 b"READY"，验证发布通道能真正送达
        reader 端：
          1. 阻塞接收 b"READY"，收到即确认通道打通
        """
        if self.is_writer:
            for _ in range(self.n_reader):
                # XPUB_VERBOSE 下，每个 reader 订阅时会发来一条订阅消息
                self._socket.recv()
            # 所有 reader 都已订阅，广播 READY 验证通道
            self._socket.send(b"READY")
        else:
            msg = self._socket.recv()
            assert msg == b"READY", f"握手失败：期望 READY，实际收到 {msg!r}"

    def enqueue(self, obj) -> None:
        """writer 端：广播一个对象（pickle 序列化后发送）"""
        assert self.is_writer, "只有 writer 能 enqueue"
        self._socket.send(pickle.dumps(obj))

    def dequeue(self, timeout: float | None = None):
        """reader 端：接收一个对象（反序列化后返回）

        Args:
            timeout: 秒。超时抛 TimeoutError；None 表示无限阻塞。
        """
        assert not self.is_writer, "只有 reader 能 dequeue"
        if timeout is not None:
            # poll 支持超时，避免无限阻塞（便于优雅关闭）
            if not self._socket.poll(timeout=int(timeout * 1000)):
                raise TimeoutError("MessageQueue.dequeue 超时")
        return pickle.loads(self._socket.recv())

    def shutdown(self) -> None:
        """关闭 socket 和 context"""
        self._socket.close()
        self._context.term()
