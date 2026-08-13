# 03 · EngineCore 时序图

> **EngineCoreProc 的 busy loop 一个完整迭代**：收请求 → 调度 → 执行 → 回传结果。

```mermaid
sequenceDiagram
    autonumber
    participant FE as 引擎前端<br/>(AsyncLLM)
    participant ZI as ZMQ 收线程<br/>(process_input_sockets)
    participant ECP as EngineCoreProc<br/>(busy loop)
    participant SC as Scheduler
    participant EX as Executor
    participant ZO as ZMQ 发线程<br/>(process_output_sockets)

    FE->>ZI: ZMQ 发送 (ADD, EngineCoreRequest)
    ZI->>ZI: MsgpackDecoder 反序列化
    ZI->>ECP: input_queue.put((ADD, req))

    loop run_busy_loop 每次迭代
        ECP->>ECP: _process_input_queue()
        ECP->>ECP: input_queue.get() → _handle_client_request()
        alt 请求类型 = ADD
            ECP->>SC: add_request(req) 进入 waiting 队列
        else 请求类型 = ABORT
            ECP->>SC: abort_requests(ids)
        end

        ECP->>ECP: _process_engine_step()
        ECP->>ECP: step_fn() = step()
        ECP->>SC: schedule() → SchedulerOutput
        SC-->>ECP: 本轮选中的请求 + KV 块
        ECP->>EX: execute_model(scheduler_output, non_block=True)
        EX-->>ECP: future
        ECP->>ECP: future.result() 等 GPU 出 logits
        ECP->>EX: sample_tokens() (采样出 token)
        EX-->>ECP: ModelRunnerOutput
        ECP->>SC: update_from_output(scheduler_output, output)
        SC-->>ECP: EngineCoreOutputs (request_id → 新 token)
        ECP->>ECP: post_step() (投机解码草稿 token)
    end

    ECP->>ZO: output_queue.put((engine_idx, EngineCoreOutputs))
    ZO->>ZO: MsgpackEncoder 序列化
    ZO->>FE: ZMQ 发送结果
```

## 关键点

1. **busy loop 三步**：`_process_input_queue()`（收请求）→ `_process_engine_step()`（推理）→
   `post_step()`。见 `core.py:1585`。
2. **执行是异步的**：`execute_model(non_block=True)` 立刻返回 `future`，`future.result()`
   才真正阻塞等 GPU。这为 PP 流水线（batch queue）埋下伏笔。
3. **调度器输出是"半成品"**：`scheduler.schedule()` 只选出请求 + 分配 KV 块；
   真正的 token 是 `sample_tokens()` 采出来的，再交给 `update_from_output()` 更新请求状态。
4. **收发都是独立线程**：busy loop 主线程不碰 ZMQ，全靠两个 IO 线程 `put`/`get` 队列，
   避免序列化/反序列化阻塞调度循环。
