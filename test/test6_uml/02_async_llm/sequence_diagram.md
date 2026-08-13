# 02 · AsyncLLM 时序图

> 一条请求的两条并行线：**主线** `generate()` 负责入队，**后台线** `output_handler` 负责拉结果回填。
> 两条线通过 `RequestOutputCollector`（asyncio.Queue）解耦。

```mermaid
sequenceDiagram
    autonumber
    participant R as Router<br/>(api_server)
    participant G as AsyncLLM.generate()
    participant IP as InputProcessor
    participant OP as OutputProcessor
    participant EC as EngineCoreClient<br/>(ZMQ 传输层)
    participant OH as output_handler<br/>(后台协程)
    participant Q as RequestOutputCollector<br/>(asyncio.Queue)

    Note over G,OH: ===== 主线：入队请求 =====
    R->>G: generate(prompt, params, request_id)
    G->>IP: process_inputs_async(prompt, params)
    IP-->>G: EngineCoreRequest (tokenize/多模态完成)
    G->>OP: add_request(request, ..., queue)
    Note over OP: 前端先登记请求，为后面回填做准备
    G->>EC: add_request_async(request)
    EC-->>G: (请求已发给后端进程)
    Note over G: 创建 Q，把请求绑定到 Q

    par 后台协程持续拉结果
        loop 只要后端有输出
            OH->>EC: get_output_async()
            EC-->>OH: EngineCoreOutputs
            OH->>OP: process_outputs(outputs_slice)
            OP-->>OH: 拆分出的 RequestOutput
            OH->>Q: q.put(RequestOutput)
        end
    and 主线迭代输出
        loop 直到 finished
            G->>Q: q.get() / get_nowait()
            Q-->>G: RequestOutput(token)
            G-->>R: yield RequestOutput
        end
    end

    G-->>R: finished=True
    G->>Q: q.close()
```

## 关键点

1. **入队与出队分离**：`generate()` 只负责把请求 `add_request` 进去，然后开始 `await q.get()`；
   真正拉后端结果的是后台 `output_handler` 协程，两者靠 `RequestOutputCollector` 解耦。
2. **`n > 1` 采样会 fan-out**：`add_request()` 里若 `params.n > 1`，会构造 `ParentRequest`，
   把请求复制成 n 个 child request 分别入队，最后再由 OutputProcessor 合并。
3. **断连即中止**：`generate()` 被 `CancelledError`（客户端断连）时，会 `await self.abort()` 中止该请求。
4. **分片处理防阻塞**：`output_handler` 用 `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` 把一大包输出切成小片
   处理，片间 `await asyncio.sleep(0)` 让出事件循环。
