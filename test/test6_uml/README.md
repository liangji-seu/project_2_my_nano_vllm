# vLLM V1 全链路 UML 图集

从 API Server（FastAPI）一路到 Worker 内部的 GPU 推理，按模块拆分，每个模块一个文件夹，
包含一张**类图（class_diagram.md）**和一张**时序图（sequence_diagram.md）**。

## 模块目录（按调用顺序）

| # | 目录 | 模块 | 核心源码 |
|---|------|------|----------|
| 01 | [01_api_server](./01_api_server) | FastAPI 入口 | `vllm/entrypoints/openai/api_server.py` |
| 02 | [02_async_llm](./02_async_llm) | 引擎前端（AsyncLLM） | `vllm/v1/engine/async_llm.py` |
| 03 | [03_engine_core](./03_engine_core) | 引擎后端（EngineCore/EngineCoreProc） | `vllm/v1/engine/core.py` |
| 04 | [04_scheduler](./04_scheduler) | 调度器（Scheduler） | `vllm/v1/core/sched/scheduler.py` |
| 05 | [05_executor](./05_executor) | 执行器（MultiprocExecutor） | `vllm/v1/executor/multiproc_executor.py` |
| 06 | [06_worker](./06_worker) | Worker 启动与工作 | `vllm/v1/worker/worker_base.py` / `gpu_worker.py` |
| 07 | [07_kv_cache_manager](./07_kv_cache_manager) | KV Cache 管理 | `vllm/v1/core/kv_cache_manager.py` |
| 08 | [08_worker_internal](./08_worker_internal) | ModelRunner 内部 | `vllm/v1/worker/gpu_model_runner.py` |
| 09 | [09_startup_rpc](./09_startup_rpc) | Executor→Worker→ModelRunner→ModelLoader 启动与 RPC | `multiproc_executor.py` / `worker_base.py` / `gpu_worker.py` / `model_loader/` |

## 端到端数据流一图流

```
HTTP 请求
  → FastAPI router (01)
  → AsyncLLM.generate() 输入处理 + 后台 output_handler (02)
  → [ZMQ] EngineCoreProc.run_busy_loop (03)
      → Scheduler.schedule() 选请求 + 分配 KV cache (04)
      → MultiprocExecutor.collective_rpc("execute_model") (05)
          → Worker 子进程 busy loop 查表执行 (06)
              → GPUModelRunner.execute_model() 前向 + 采样 (08)
          → KVCacheManager.allocate_slots() 块分配 (07)
  → 结果沿 output_queue/ZMQ 回传 → detokenize → 流式返回
```
