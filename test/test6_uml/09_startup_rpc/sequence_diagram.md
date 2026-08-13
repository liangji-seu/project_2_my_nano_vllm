# 09 · 从 EngineCore 构造到 run_busy_loop 的完整启动时序图

> 三种颜色背景 = 三个 OS 进程：
> 🟦 蓝 = 引擎前端 · 🟩 绿 = 引擎后端 · 🟧 橙 = Worker 子进程 · 🟥 红 = 跨进程握手/连接。

```mermaid
sequenceDiagram
    autonumber
    participant FE as 🟦引擎前端<br/>AsyncLLM
    participant EC as 🟩EngineCoreProc<br/>(后端进程)
    participant ME as 🟩MultiprocExecutor
    participant SC as 🟩Scheduler
    participant SOM as 🟩Structured<br/>OutputManager
    participant WP as 🟧WorkerProc<br/>(子进程)
    participant WW as 🟧Worker<br/>WrapperBase
    participant W as 🟧Worker<br/>(gpu_worker)
    participant MR as 🟧GPUModelRunner
    participant ML as 🟧ModelLoader

    rect rgb(224, 238, 255)
    Note over FE: 进程① 引擎前端
    FE->>FE: EngineCoreClient.make_async_mp_client()
    FE->>EC: spawn 子进程, 入口 run_engine_core()
    end

    rect rgb(235, 255, 235)
    Note over EC: 进程② 引擎后端 —— EngineCoreProc.__init__()
    EC->>EC: input_queue / output_queue 创建
    EC->>EC: identity = engine_index 的字节序
    Note over EC: executor_fail_callback = 向 input_queue 塞 EXECUTOR_FAILED
    end

    rect rgb(255, 235, 238)
    Note over FE,EC: ⚡ 跨进程握手①：前端 ↔ 后端 (ZMQ)
    EC->>FE: _perform_handshakes()<br/>DEALER/ROUTER 交换 input/output 地址
    FE-->>EC: 确认握手, 返回 addresses
    end

    rect rgb(235, 255, 235)
    Note over EC: 进入 super().__init__() = EngineCore.__init__()
    EC->>EC: load_general_plugins()
    EC->>ME: model_executor = executor_class(vllm_config)
    Note over ME: = MultiprocExecutor 构造<br/>→ 触发 _init_executor()
    ME->>ME: get_distributed_init_method()<br/>→ tcp://127.0.0.1:port
    ME->>ME: rpc_broadcast_mq = MessageQueue(world_size, ...)
    ME->>ME: scheduler_output_handle = export_handle()
    end

    rect rgb(255, 235, 238)
    Note over ME,WP: ⚡ 跨进程握手②：后端 → Worker (spawn)
    loop 每个 local_rank
        ME->>WP: make_worker_process() → proc.start()<br/>传 input_shm_handle / ready_pipe / death_pipe
    end
    end

    rect rgb(255, 243, 224)
    Note over WP: 进程③ Worker 子进程 —— worker_main()
    WP->>WW: WorkerWrapperBase(rpc_rank, global_rank)
    WP->>WW: init_worker(all_kwargs)
    WW->>WW: resolve_obj_by_qualname(worker_cls)
    WW->>W: self.worker = worker_class(**kwargs)
    W->>W: Worker.__init__() 记录 device/rank/config

    Note over WP,MR: init_device: 设卡 + 造 runner
    WP->>WW: init_device()
    WW->>W: self.worker.init_device()
    W->>W: torch.accelerator.set_device_index() 绑 GPU
    W->>W: init_worker_distributed_environment() 拉 NCCL
    W->>W: MemorySnapshot() 测量显存
    W->>MR: self.model_runner = GPUModelRunnerV1(config, device)

    Note over WP,ML: load_model: 加载权重
    WP->>WW: load_model()
    WW->>W: self.worker.load_model()
    W->>MR: model_runner.load_model()
    MR->>ML: get_model_loader(load_config)
    ML->>ML: load_model(vllm_config, model_config)
    ML-->>MR: self.model = nn.Module

    Note over WP: 建立 IPC 消息队列
    WP->>WP: _init_message_queues(input_shm_handle)
    Note over WP: rpc_broadcast_mq = create_from_handle()<br/>worker_response_mq = MessageQueue(1,1)
    end

    rect rgb(255, 235, 238)
    Note over WP,ME: ⚡ 跨进程握手③：Worker → 后端 (READY)
    WP->>ME: ready_writer.send(READY + response_mq 的 handle)
    ME->>ME: wait_for_ready() 收到 READY
    ME->>ME: create_from_handle() 连上 worker_response_mq
    ME->>ME: rpc_broadcast_mq.wait_until_ready()
    ME-->>EC: 所有 worker 就绪, 返回 self.workers
    end

    rect rgb(235, 255, 235)
    Note over EC: 回到 EngineCore.__init__ 继续构造
    EC->>EC: _initialize_kv_caches(vllm_config)<br/>(profiling 显存 → 定 block 数)
    EC->>SOM: structured_output_manager =<br/>StructuredOutputManager(vllm_config)
    EC->>SC: Scheduler = get_scheduler_cls()
    EC->>SC: scheduler = Scheduler(vllm_config, kv_cache_config,<br/>structured_output_manager, block_size, ...)
    SC-->>EC: scheduler 就绪 (含 kv_cache_manager)
    EC->>EC: request_block_hasher (前缀缓存 hash)
    EC->>EC: step_fn = step / step_with_batch_queue
    end

    rect rgb(235, 255, 235)
    Note over EC: 启动 ZMQ IO 线程
    EC->>EC: input_thread = Thread(process_input_sockets)
    EC->>EC: output_thread = Thread(process_output_sockets)
    Note over EC: 两个 IO 线程在队列与 ZMQ 间搬运
    end

    rect rgb(235, 255, 235)
    Note over EC: run_engine_core → 进入主循环
    EC->>EC: run_busy_loop()
    loop 每次迭代
        EC->>EC: _process_input_queue() 收请求
        EC->>SC: _process_engine_step() → scheduler.schedule()
        EC->>ME: model_executor.execute_model(scheduler_output)
        EC->>SC: scheduler.update_from_output(...)
        EC->>EC: output_queue.put(outputs)
    end
    end
```

## 关键顺序（源码核对）

1. **握手在构造 model_executor 之前**：`EngineCoreProc.__init__`（[core.py:1193](vllm/v1/engine/core.py#L1193)）
   先 `_perform_handshakes()` 跟前端完成 ZMQ 握手，之后才 `super().__init__()`（[core.py:1223](vllm/v1/engine/core.py#L1223)）。

2. **Worker 全就绪后 Scheduler 才出生**：`EngineCore.__init__` 里顺序是
   `model_executor = executor_class(...)`（**这一步 spawn 完所有 worker 并等 READY**）→
   `_initialize_kv_caches` → `StructuredOutputManager` → `Scheduler`。
   所以 `Scheduler` 构造时，worker 已经在 busy loop 里等任务了。

3. **三处握手介质不同**：
   - 前端↔后端：ZMQ DEALER/ROUTER；
   - 后端→Worker 下行：`rpc_broadcast_mq`（共享内存）+ `ready_pipe`；
   - Worker→后端回传：`worker_response_mq`。

4. **`ready_pipe` 就绪信号里带的是 `worker_response_mq` 的 handle**（[multiproc_executor.py:1162](vllm/v1/executor/multiproc_executor.py#L1162)）：
   后端拿到这个 handle 才能 `create_from_handle()` 连上回传通道——这就是"就绪"和"连接"绑在一起的原因。

5. **IO 线程最后才启动**（[core.py:1250](vllm/v1/engine/core.py#L1250)）：input/output 两个线程在
   引擎核心对象全部构造完后才起，避免构造期间就收到请求。
