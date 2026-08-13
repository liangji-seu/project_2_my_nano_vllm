# 01 · API Server（FastAPI 入口）类图

> 源码：`vllm/entrypoints/openai/api_server.py`
> 说明：这一层**没有 OOP 继承关系**，本质是一组模块级函数 + FastAPI 框架对象，
> 所以这张图更像"组件图"，画的是**谁创建谁、谁持有谁**。

```mermaid
classDiagram
    direction TB

    class FastAPI {
        +state.engine_client : EngineClient
        +state.model_config : ModelConfig
        +state.log_requests : bool
    }

    class AsyncLLM {
        <<引擎前端, 见模块02>>
    }

    class EngineClient {
        <<abstract 接口>>
    }

    class Namespace {
        <<args 命令行参数>>
        +host / port / model ...
    }

    class build_app {
        <<模块函数>>
        +build_app(args) FastAPI
        +lifespan(app)
    }

    class init_app_state {
        <<模块函数>>
        +init_app_state(engine_client, model_config, state, args)
    }

    class build_async_engine_client {
        <<模块函数>>
        +build_async_engine_client(args)
        +build_async_engine_client_from_engine_args(engine_args)
    }

    class run_server {
        <<模块函数>>
        +run_server(args, **uvicorn_kwargs)
        +setup_server(args)
    }

    class routers {
        <<注册的路由>>
        +completion / chat / embedding ...
    }

    Namespace --> build_app : 传入
    build_app ..> FastAPI : 创建 FastAPI 实例\n并注册 routers
    FastAPI o-- AsyncLLM : state.engine_client 持有
    build_async_engine_client ..> AsyncLLM : AsyncLLM.from_vllm_config() 创建
    AsyncLLM --|> EngineClient : 实现
    run_server ..> FastAPI : uvicorn.run(app)
    FastAPI o-- routers : include_router()
    init_app_state ..> FastAPI : 填充 state
```

## 关键点

1. **`build_app(args)`**：纯粹的"搭框架"——`FastAPI(lifespan=lifespan)`，
   然后把 completion / chat / embedding 等一堆 router `include_router()` 挂上去。
   - `build_app` 里只做**路由注册**，不碰模型。
2. **`lifespan` 上下文**：FastAPI 启动时先进 `lifespan`，在 `yield` 之前调
   `init_app_state()`，用 `build_async_engine_client()` 真正把 `AsyncLLM` 造出来，
   塞进 `app.state.engine_client`，供所有 handler 使用。
3. **`build_async_engine_client`**：`AsyncEngineArgs.from_cli_args(args)` →
   `create_engine_config()` → `AsyncLLM.from_vllm_config()`。这里才是"引擎前端"的出生点。
4. **`run_server`**：最终 `uvicorn.run(...)`，把 `build_app` 造出来的 app 跑起来。
