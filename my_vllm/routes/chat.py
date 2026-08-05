"""
OpenAI 兼容 Chat Completions 接口（占位实现）

POST /v1/chat/completions → 流式返回模拟 token
"""

import asyncio
import json
import time

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import StreamingResponse

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI 兼容的 /v1/chat/completions 接口

    当前为占位实现，返回模拟的流式 token 序列。
    后续对接真实引擎后替换为真正推理。
    """
    # 读取请求体
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", request.app.state.served_model_name)

    # 提取最后一条用户消息
    user_prompt = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_prompt = msg.get("content", "")

    chat_id = f"chatcmpl-{int(time.time())}"

    async def generate():
        """逐 token 流式输出（SSE 格式）"""
        tokens = [
            "你好",
            "！",
            "我是",
            "my_vllm",
            "，",
            "一个",
            "极简",
            "推理",
            "服务",
            "。",
        ]
        for i, token in enumerate(tokens):
            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.2)  # 模拟推理延迟

        # 结束标记
        final_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def attach_router(app: FastAPI):
    app.include_router(router)
