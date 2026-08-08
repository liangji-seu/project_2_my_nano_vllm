"""
引擎层

前端:
  EngineClient  — 协议（Protocol），定义引擎前端的接口契约
  AsyncLLM      — 异步引擎前端实现

后端:
  EngineCore     — 纯推理逻辑（预留: scheduler + executor + KV cache）
  EngineCoreProc — 子进程包装 + ZMQ 通信层

传输层:
  CoreEngineProcManager — 引擎后端进程管理器
  launch_core_engines   — 启动引擎进程的 contextmanager
"""
