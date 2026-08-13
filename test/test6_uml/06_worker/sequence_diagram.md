# 06 · Worker 时序图

> Worker 子进程的**生命周期**：初始化 → 进 busy loop → 收到 RPC 查表执行。

```mermaid
sequenceDiagram
    autonumber
    participant MP as MultiprocExecutor<br/>(父进程)
    participant WP as WorkerProc<br/>(子进程)
    participant WW as WorkerWrapperBase
    participant W as Worker<br/>(gpu_worker)
    participant MR as GPUModelRunner

    Note over MP,MR: ===== 启动期 =====
    MP->>WP: spawn 子进程, 传入 vllm_config + Handle
    WP->>WW: init_worker(all_kwargs)
    WW->>W: worker_class(**kwargs) 构造真 Worker
    W->>MR: 创建 GPUModelRunner
    WW-->>WP: worker 就绪
    WP->>W: init_device()
    W->>W: torch.distributed.init_process_group()
    W->>W: 分配 GPU / 设置 CUDA 环境
    WP->>W: load_model()
    W->>W: 加载权重到显存
    WP->>WP: _init_message_queues() 连接 IPC 队列
    WP-->>MP: ready_pipe.send("READY")
    Note over WP: 进入 worker_busy_loop()

    Note over MP,MR: ===== 运行期：一次 execute_model =====
    MP->>WP: rpc_broadcast_mq 里放 (method, args, ...)
    WP->>WP: busy loop dequeue 拿到四元组
    WP->>WW: func = getattr(worker, "execute_model")
    WW->>W: self.worker.execute_model(scheduler_output)
    W->>MR: model_runner.execute_model(...)
    MR-->>W: ModelRunnerOutput (采样后的 token)
    W-->>WW: output
    WW-->>WP: output
    WP->>WP: enqueue_output((SUCCESS, output))
    WP-->>MP: worker_response_mq 里放结果
```

## 关键点

1. **`init_worker` 负责"造芯"**：`worker_base.py:319` 的 `self.worker = worker_class(**kwargs)`
   这一步才真正实例化 `Worker`。
2. **`init_device` / `load_model` 各自职责**：前者建分布式进程组 + 分卡；后者加载权重。
   分开是为了支持权重更新（热加载）时不必重新初始化设备。
3. **busy loop 里"查表执行"**：`getattr(self.worker, method)` 里 `self.worker` 是 wrapper，
   靠 `__getattr__` 委托到真 Worker，所以方法名对得上就能被远程调用。
