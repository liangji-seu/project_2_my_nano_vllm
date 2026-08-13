# 06 · Worker（启动与工作）类图

> 源码：`vllm/v1/worker/worker_base.py`、`vllm/v1/worker/gpu_worker.py`
> 角色：**每个 GPU 子进程里真正干活的执行单元**。`WorkerWrapperBase` 是"壳"，
> `Worker` 是"芯"，两者靠 `__getattr__` 委托衔接。

```mermaid
classDiagram
    direction TB

    class WorkerBase {
        <<abstract 接口>>
        +init_device() None
        +load_model() None
        +execute_model(...) ModelRunnerOutput
        +sample_tokens(...)
        +determine_available_memory() int
        +shutdown() None
    }

    class WorkerWrapperBase {
        +worker : WorkerBase
        +__getattr__(attr) 委托到 self.worker
        +init_worker(kwargs) 构造真 Worker
        +init_device() None
        +load_model() None
        +execute_model(...) 套 mm_cache 后转 self.worker
    }

    class Worker {
        +model_runner : GPUModelRunner
        +init_device() None
        +load_model() None
        +execute_model(scheduler_output) ModelRunnerOutput
        +sample_tokens(...)
        +determine_available_memory() int
        +initialize_from_config() None
    }

    class WorkerProc {
        <<见模块05, 子进程外壳>>
        +worker_busy_loop()
        +getattr(self.worker, method)
    }

    class GPUModelRunner {
        <<见模块08>>
        +execute_model()
    }

    WorkerWrapperBase --|> WorkerBase : 实现
    Worker --|> WorkerBase : 实现
    WorkerWrapperBase *-- WorkerBase : self.worker 持有真 Worker
    WorkerProc *-- WorkerWrapperBase : self.worker 指向 wrapper
    Worker *-- GPUModelRunner : model_runner 组合
```

## 关键点

1. **两层壳**：
   - `WorkerProc`（模块 05）里的 `self.worker` 其实是 `WorkerWrapperBase`；
   - `WorkerWrapperBase` 里的 `self.worker` 才是真正的 `Worker`（gpu_worker.py）。
2. **`__getattr__` 委托**（`worker_base.py:333`）：`WorkerWrapperBase` 本身不实现所有方法，
   遇到不认识的方法就 `return getattr(self.worker, attr)`，所以
   `getattr(wrapper, "execute_model")` 能一路查到真 Worker。
3. **`execute_model` 特殊处理**（`worker_base.py:346`）：wrapper 的 `execute_model` 先套一层
   多模态缓存（`_apply_mm_cache`），再调 `self.worker.execute_model`——这是壳能"增强"芯的地方。
4. **启动链路**：`init_worker`（构造 Worker）→ `init_device`（分配 GPU）→ `load_model`（加载权重），
   都在子进程里按顺序执行。
