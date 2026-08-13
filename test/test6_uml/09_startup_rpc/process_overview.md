# 09 · 三个进程 + 对象归属总览图

> 用 `flowchart` 的 `subgraph` 表示**三个 OS 进程**（不同背景色），
> 每个 subgraph 内列该进程持有的对象，箭头表示**跨进程连接**。

```mermaid
flowchart TB
    subgraph FE["进程① 引擎前端 (AsyncLLM)"]
        direction TB
        AL["AsyncLLM<br/>EngineClient"]
        MQ1["MPClient<br/>ZMQ 传输层"]
    end

    subgraph EC["进程② 引擎后端 (EngineCoreProc)"]
        direction TB
        CORE["EngineCore"]
        SCH["Scheduler"]
        SOM["StructuredOutputManager"]
        ME["MultiprocExecutor"]
        ZI["ZMQ 输入线程<br/>process_input_sockets"]
        ZO["ZMQ 输出线程<br/>process_output_sockets"]
        BMQ["rpc_broadcast_mq<br/>(MessageQueue 广播)"]
    end

    subgraph W["进程③ Worker 子进程 (× N 卡)"]
        direction TB
        WP["WorkerProc"]
        WW["WorkerWrapperBase"]
        GW["Worker (gpu_worker)"]
        MR["GPUModelRunner"]
        ML["ModelLoader"]
        RMQ["worker_response_mq"]
    end

    AL --> MQ1
    MQ1 -->|"ZMQ DEALER/ROUTER"| ZI
    ZO -->|"ZMQ"| MQ1
    CORE --> SCH
    CORE --> SOM
    CORE --> ME
    ME --> BMQ
    ME -->|"spawn + ready_pipe"| WP
    BMQ -->|"共享内存 / XPUB"| WP
    WP --> WW
    WW --> GW
    GW --> MR
    MR --> ML
    WP --> RMQ
    RMQ -->|"回传结果"| ME
    ZI -->|"input_queue"| CORE
    CORE -->|"output_queue"| ZO

    style FE fill:#e2efff,stroke:#4c8bf5
    style EC fill:#e6ffe6,stroke:#4caf50
    style W fill:#fff3e0,stroke:#ff9800
```

## 三个进程一句话

| 进程 | 颜色 | 职责 | 与谁通信 |
|------|------|------|----------|
| ① 引擎前端 | 蓝 | 收 HTTP、预处理输入、回传 token | 进程②（ZMQ） |
| ② 引擎后端 | 绿 | 调度 + 执行 + KV cache 账本 | 进程①（ZMQ）、进程③（共享内存 + pipe） |
| ③ Worker | 橙 | GPU 前向 + 采样 | 进程②（共享内存 + pipe） |

## 跨进程连接三种介质

1. **前端 ↔ 后端**：ZMQ `DEALER/ROUTER`（输入线程 + 输出线程收发包）。
2. **后端 ↔ Worker（下行业务数据）**：`rpc_broadcast_mq` —— 共享内存环形缓冲区，大数据溢出到 XPUB。
3. **后端 ↔ Worker（握手/存活/回传）**：`ready_pipe`（就绪）、`death_pipe`（父死检测）、`worker_response_mq`（结果回传）。
