# 05 · MultiprocExecutor 时序图

> 上半张：**启动期**（spawn 子进程 → 就绪 → 进入 busy loop）。
> 下半张：**运行期**（`execute_model` 一次完整往返）。

```mermaid
sequenceDiagram
    autonumber
    participant E as MultiprocExecutor<br/>(父进程)
    participant MQ as rpc_broadcast_mq<br/>(ShmRingBuffer + XPUB)
    participant SC as SpinCondition<br/>(notify socket)
    participant P as WorkerProc<br/>(子进程)
    participant WW as WorkerWrapperBase

    Note over E,P: ===== 启动期 =====
    E->>E: _init_executor()
    E->>E: 创建 rpc_broadcast_mq 并 export_handle()
    E->>P: spawn 子进程, 传入 Handle
    P->>WW: init_worker() → 构造真 Worker
    P->>WW: init_device() / load_model()
    P->>P: _init_message_queues() 连接共享内存/ZMQ
    P-->>E: ready_pipe.send("READY")
    E->>E: wait_for_ready() 收齐所有 READY
    Note over P: 进入 worker_busy_loop()<br/>阻塞在 rpc_broadcast_mq.dequeue()

    Note over E,WW: ===== 运行期：execute_model =====
    E->>E: collective_rpc("execute_model", args=(scheduler_output,))
    E->>MQ: enqueue(("execute_model", args, kwargs, output_rank))
    Note over MQ: 小数据→写共享内存<br/>大数据→XPUB 直接发
    E->>SC: _spin_condition.notify() (PUB 1 字节)
    SC-->>P: SUB 收到唤醒 ping
    P->>MQ: dequeue() 读共享内存
    MQ-->>P: (method, args, kwargs, output_rank)
    P->>WW: func = getattr(worker, "execute_model")
    WW->>WW: func(*args, **kwargs) 真 Worker 前向推理
    WW-->>P: output (ModelRunnerOutput)
    P->>P: handle_output() → enqueue_output(SUCCESS, output)
    P->>MQ: 结果 enqueue 到 worker_response_mq

    Note over E: 调用方 future.result() 才真正收结果
    E->>MQ: dequeue() 收结果
    MQ-->>E: (SUCCESS, output)
    E-->>E: 返回 ModelRunnerOutput
```

## 关键点

1. **发出去 ≠ 等回来**：`collective_rpc` 只 `enqueue` 然后立刻返回 `FutureWrapper`，不阻塞。
2. **数据与信号分离**：数据走 `rpc_broadcast_mq`（共享内存 / XPUB），唤醒走 `SpinCondition`
   的 notify socket（1 字节 ping）。
3. **查表执行**：Worker 侧没有 if/elif 分发表，就是 `getattr(self.worker, method)(*args, **kwargs)`。
4. **结果靠"拉"**：结果不是 Worker "推"回来的，而是 Executor 侧 `future.result()` 主动
   `dequeue()` 拉回来；`FutureWrapper` 的 FIFO drain 保证多并发 RPC 按发出顺序返回。
