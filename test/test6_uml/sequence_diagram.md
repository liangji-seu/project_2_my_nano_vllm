# MultiprocExecutor — Worker 时序图（UML Sequence Diagram）

> 在 VSCode 安装 `Markdown Preview Mermaid Support` 插件后即可预览。
> `->>` 同步调用(带返回) · `-->>` 异步消息 · `Note` 注释。

## 一、启动时序（spawn → 就绪 → busy loop）

```mermaid
sequenceDiagram
    autonumber
    participant E as MultiprocExecutor<br/>(父进程)
    participant P as WorkerProc<br/>(子进程)
    participant WW as WorkerWrapperBase
    participant W as Worker(gpu_worker)

    E->>E: _init_executor()
    E->>E: 创建 rpc_broadcast_mq (writer 端)<br/>并 export_handle() 导出连接信息
    E->>P: spawn 子进程, 通过参数传入 Handle
    activate P
    P->>WW: init_worker(kwargs) → 构造真 Worker
    P->>P: torch.distributed.init_process_group()
    P->>WW: init_device() → 分配 GPU
    P->>WW: load_model() → 加载权重
    P->>P: _init_message_queues()<br/>create_from_handle() 连接共享内存/ZMQ
    P-->>E: ready_pipe.send("READY") + response_mq 的 handle
    deactivate P
    E->>E: wait_for_ready() 收到所有 READY
    Note over P: 进入 worker_busy_loop()<br/>阻塞在 rpc_broadcast_mq.dequeue()
```

## 二、RPC 运行时序（execute_model 一次完整往返）

```mermaid
sequenceDiagram
    autonumber
    participant S as Scheduler<br/>(EngineCore)
    participant E as MultiprocExecutor
    participant BMQ as rpc_broadcast_mq<br/>(ShmRingBuffer+XPUB)
    participant SC as SpinCondition<br/>(notify socket)
    participant P as WorkerProc<br/>(busy loop)
    participant WW as WorkerWrapperBase
    participant RMQ as worker_response_mq

    Note over P: 已在 dequeue() 阻塞等待

    S->>E: execute_model(scheduler_output)
    E->>E: collective_rpc("execute_model", args=(scheduler_output,), non_block)
    E->>BMQ: enqueue(("execute_model", args, kwargs, output_rank))
    Note over BMQ: 小数据→写共享内存环形缓冲区<br/>大数据(≥max_chunk_bytes)→XPUB直接发
    BMQ->>SC: _spin_condition.notify()
    SC-->>P: PUB 发 1 字节 b'\x00' 唤醒
    P->>BMQ: dequeue() 被唤醒, 读共享内存
    BMQ-->>P: (method, args, kwargs, output_rank)
    P->>WW: func = getattr(worker, "execute_model")
    WW->>WW: func(*args, **kwargs)<br/>→ _apply_mm_cache → 真 Worker.execute_model()
    WW-->>P: output (ModelRunnerOutput)
    P->>P: handle_output() → enqueue_output()<br/>打包成 (SUCCESS, output)
    P->>RMQ: enqueue((SUCCESS, output))

    Note over E: 调用方 future.result() 才真正收结果
    E->>RMQ: dequeue() 收结果
    RMQ-->>E: (SUCCESS, output)
    E-->>S: 返回 ModelRunnerOutput
```

## 关键点说明

1. **发送侧（第 4-5 步）**：`collective_rpc` 只做一件事——把 `(method, args, kwargs, output_rank)`
   这个四元组 `enqueue` 进广播队列，**立刻返回一个 `FutureWrapper`**，不阻塞等结果。
   这正是异步 RPC 的核心：发出去 ≠ 等回来。

2. **数据与信号的分离**：
   - 数据走 `rpc_broadcast_mq`（共享内存，或大数据溢出到 XPUB）
   - 唤醒走 `SpinCondition` 的 notify socket（普通 PUB/SUB，1 字节 ping）

3. **查表执行（第 8-9 步）**：Worker 侧没有 if/elif 分发表，就一行
   `getattr(self.worker, method)(*args, **kwargs)`，方法名对得上就能被远程调用。

4. **收结果（第 12-14 步）**：结果不是 Worker "推"回来的，而是 Executor 侧
   `future.result()` 主动去 `worker_response_mq` 里 `dequeue()` 拉回来的；
   `FutureWrapper` 用 FIFO drain 保证多个并发 RPC 的结果按发出顺序返回。
