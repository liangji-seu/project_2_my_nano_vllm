# 03 · EngineCore（引擎后端）类图

> 源码：`vllm/v1/engine/core.py`
> 角色：**引擎后端**。`EngineCore` 是纯计算核心（调度 + 执行）；
> `EngineCoreProc` 给它套上"进程 + ZMQ 收发线程 + busy loop"的外壳。

```mermaid
classDiagram
    direction TB

    class EngineCore {
        +vllm_config : VllmConfig
        +model_executor : Executor
        +scheduler : SchedulerInterface
        +structured_output_manager
        +step_fn : Callable
        +aborts_queue : Queue
        +add_request(request)
        +abort_requests(ids)
        +step() (outputs, model_executed)
        +post_step(model_executed)
        +shutdown()
    }

    class EngineCoreProc {
        +input_queue : Queue
        +output_queue : Queue
        +aborts_queue : Queue
        +shutdown_state : EngineShutdownState
        +run_busy_loop() None
        +_process_input_queue() None
        +_process_engine_step() bool
        +_handle_client_request(req) None
        +process_input_sockets(...) None
        +process_output_sockets(...) None
    }

    class Scheduler {
        <<见模块04>>
    }

    class Executor {
        <<见模块05>>
    }

    class EngineShutdownState {
        <<IntEnum>>
        RUNNING / REQUESTED / ...
    }

    class ZMQInputThread {
        <<线程, process_input_sockets>>
        +DEALER socket 接收
        +MsgpackDecoder 反序列化
    }

    class ZMQOutputThread {
        <<线程, process_output_sockets>>
        +DEALER socket 发送
        +MsgpackEncoder 序列化
    }

    EngineCoreProc --|> EngineCore : 继承
    EngineCore *-- Scheduler : 组合
    EngineCore *-- Executor : model_executor 组合
    EngineCore ..> EngineShutdownState : 使用
    EngineCoreProc *-- ZMQInputThread : 收 IO 线程
    EngineCoreProc *-- ZMQOutputThread : 发 IO 线程
    ZMQInputThread ..> EngineCoreProc : 写入 input_queue
    EngineCoreProc ..> ZMQOutputThread : 读取 output_queue
```

## 关键点

1. **两层设计**：
   - `EngineCore`（纯逻辑）：`step()` = `scheduler.schedule()` → `executor.execute_model()` →
     `scheduler.update_from_output()`，不关心进程和网络。
   - `EngineCoreProc`（外壳）：`run_busy_loop()` 无限循环，两个 ZMQ IO 线程负责
     收前端请求 / 发结果，把消息塞进 `input_queue` / `output_queue`。
2. **`step_fn` 可替换**：单卡走 `self.step`；`max_concurrent_batches > 1`（PP 流水线）
   时换成 `step_with_batch_queue`。见 `core.py:285`。
3. **ZMQ 收线程（`process_input_sockets`）**：用 `zmq.DEALER` + `zmq.Poller`，
   收到 `(RequestType, RequestData)` 两帧，`MsgpackDecoder` 反序列化后 `input_queue.put_nowait()`。
4. **abort 双队列**：ABORT 请求同时进 `aborts_queue` 和 `input_queue`，既能急切处理，又保证顺序。
