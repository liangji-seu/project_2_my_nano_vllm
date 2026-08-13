# MultiprocExecutor — Worker 类关系图（UML Class Diagram）

> 在 VSCode 安装 `Markdown Preview Mermaid Support` 插件后即可预览。
> 箭头含义：`<|--` 继承 · `*--` 组合(强拥有) · `o--` 聚合(弱拥有) · `..>` 依赖。

```mermaid
classDiagram
    class Executor {
        <<abstract>> 基类
        +vllm_config
        +collective_rpc(method)*
        +execute_model()*
    }
    class MultiprocExecutor {
        +workers: list~WorkerProcHandle~
        +rpc_broadcast_mq: MessageQueue
        +response_mqs: list~MessageQueue~
        +futures_queue: deque~FutureWrapper~
        +output_rank: int
        +collective_rpc(method, args, kwargs)
        +execute_model(scheduler_output)
        +sample_tokens(grammar_output)
        +shutdown()
    }
    class FutureWrapper {
        +futures_queue: deque
        +get_response: Callable
        +result()
        +_wait_for_response()
    }
    class Future {
        <<concurrent.futures>>
    }
    class WorkerProcHandle {
        +proc: BaseProcess
        +rank: int
        +worker_response_mq: MessageQueue
    }
    class WorkerProc {
        <<子进程入口>>
        +rank: int
        +worker: WorkerWrapperBase
        +rpc_broadcast_mq: MessageQueue
        +worker_response_mq: MessageQueue
        +worker_busy_loop()
        +handle_output()
        +enqueue_output()
    }
    class WorkerWrapperBase {
        +worker: WorkerBase
        +rpc_rank: int
        +init_worker(kwargs)
        +init_device()
        +execute_model()
    }
    class WorkerBase {
        <<abstract>> 硬件无关接口
        +init_device()*
        +load_model()*
        +execute_model()*
        +sample_tokens()*
        +shutdown()
    }
    class Worker {
        <<gpu_worker.Worker>>
        +model_runner
        +device
        +execute_model()
        +init_device()
        +load_model()
    }
    class MessageQueue {
        +buffer: ShmRingBuffer
        +local_socket: XPUB
        +remote_socket: XPUB
        +_spin_condition: SpinCondition
        +enqueue(obj)
        +dequeue()
        +export_handle()
        +create_from_handle()
    }
    class ShmRingBuffer {
        +max_chunk_bytes: int
        +max_chunks: int
        +acquire_write()
        +acquire_read()
    }
    class SpinCondition {
        +local_notify_socket: PUB/SUB
        +busy_loop_s: float
        +wait()
        +notify()
        +cancel()
    }
    class Handle {
        <<dataclass>> 连接说明书
        +buffer_handle
        +local_subscribe_addr
        +local_notify_addr
        +remote_subscribe_addr
    }

    Executor <|-- MultiprocExecutor : 继承
    Future <|-- FutureWrapper : 继承
    WorkerBase <|-- Worker : 继承

    MultiprocExecutor *-- WorkerProcHandle : workers
    MultiprocExecutor *-- MessageQueue : rpc_broadcast_mq
    MultiprocExecutor *-- MessageQueue : response_mqs
    MultiprocExecutor *-- FutureWrapper : futures_queue
    WorkerProcHandle o-- MessageQueue : worker_response_mq

    WorkerProc *-- WorkerWrapperBase : worker
    WorkerWrapperBase *-- WorkerBase : worker

    MessageQueue *-- ShmRingBuffer : buffer
    MessageQueue *-- SpinCondition : _spin_condition
    MessageQueue ..> Handle : create_from_handle
```

## 关键点说明

1. **两层 "worker" 别搞混**：
   - `WorkerProc.worker` 是 `WorkerWrapperBase`（包装器）
   - `WorkerWrapperBase.worker` 才是真正的 `Worker`（`gpu_worker.Worker`，继承 `WorkerBase`）
   - 因此 `worker_busy_loop` 里 `getattr(self.worker, method)` 拿到的是**包装器**的方法：
     包装器自己定义了 `execute_model`（会先 `_apply_mm_cache` 再调真 Worker）；
     没定义的方法则靠 `WorkerWrapperBase.__getattr__` 转发给内层真 Worker（见
     [worker_base.py:333](vllm/vllm/v1/worker/worker_base.py#L333)）。

2. **进程边界**：`MultiprocExecutor`（父进程）与 `WorkerProc`（子进程）之间没有直接引用，
   唯一的"桥"是 `Handle` + `MessageQueue`（共享内存 + XPUB + notify socket）。
   `WorkerProcHandle` 是父进程手里的一张"名片"，只记录子进程的 `proc` 和它的回传队列。

3. **`FutureWrapper` 继承标准库 `Future`**：让 `execute_model` 在 `non_block=True` 时
   立即返回一个 Future，`result()` 时才按 FIFO 顺序真正去 `response_mqs` 收结果。
