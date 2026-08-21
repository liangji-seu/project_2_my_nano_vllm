"""
OpenAI 兼容 Chat Completions 接口

POST /v1/chat/completions → 通过 ZMQ 发送到引擎后端 → 返回完整回复

后续 commit 实现流式 SSE（逐 token）
"""

import json
import time

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI 兼容的 /v1/chat/completions 接口

    流程:
      1. 解析请求中的 messages
      2. 拼接成 prompt 字符串
      3. 通过 AsyncLLM.generate() → ZMQ → 引擎后端 → 返回结果
      4. 返回 JSON 格式响应（非流式）

    后续 commit 实现:
      - 流式 SSE（逐 token）
      - tokenizer 预处理（apply chat template）
    """
    request_start_ns = time.perf_counter_ns()
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", request.app.state.served_model_name)
    max_tokens = body.get("max_tokens", 100)
    return_metrics = bool(body.get("return_metrics", False))
    ignore_eos = bool(body.get("ignore_eos", False))

    # 提取用户消息拼接为 prompt
    # 后续 commit: 使用 tokenizer.apply_chat_template()
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt_parts.append(f"[{role}]: {content}")
    prompt = "\n".join(prompt_parts)

    # 通过引擎前端发送推理请求
    engine_client = request.app.state.engine_client
    generation_result = await engine_client.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        return_metrics=return_metrics,
        request_start_ns=request_start_ns,
        ignore_eos=ignore_eos,
    )
    if isinstance(generation_result, dict):
        generated_text = generation_result["text"]
        benchmark_metrics = generation_result.get("benchmark_metrics")
    else:
        generated_text = generation_result
        benchmark_metrics = None

    # 构造 OpenAI 兼容响应
    chat_id = f"chatcmpl-{int(time.time())}"
    response = {
        "id": chat_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": len(generated_text),
            "total_tokens": len(prompt) + len(generated_text),
        },
    }
    if benchmark_metrics is not None:
        response["benchmark_metrics"] = benchmark_metrics
    return JSONResponse(content=response)


def attach_router(app: FastAPI):
    app.include_router(router)
