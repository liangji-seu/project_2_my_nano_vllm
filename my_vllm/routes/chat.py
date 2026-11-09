# OpenAI-compatible chat completion route with request observability.

import json
import logging
import time
import uuid

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter()
logger = logging.getLogger(__name__)


def prepare_chat_prompt(messages: list[dict], tokenizer) -> tuple[str, list[int], int]:
    if not messages:
        raise ValueError("messages must contain at least one message")
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_token_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    raw_content_tokens = 0
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError("only text message content is supported")
        raw_content_tokens += len(tokenizer.encode(content, add_special_tokens=False))
    return formatted_prompt, list(prompt_token_ids), raw_content_tokens


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", request.app.state.served_model_name)
    max_tokens = body.get("max_tokens", 100)
    request_id = uuid.uuid4().hex

    try:
        prompt, prompt_token_ids, raw_content_tokens = prepare_chat_prompt(
            messages, request.app.state.chat_tokenizer
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("request_id=%s chat template rendering failed", request_id)
        raise HTTPException(status_code=500, detail="chat template rendering failed") from exc

    max_model_len = request.app.state.vllm_config.max_model_len
    logger.info(
        "request_id=%s raw_messages=%s raw_content_tokens=%d",
        request_id,
        json.dumps(messages, ensure_ascii=False, separators=(",", ":")),
        raw_content_tokens,
    )
    logger.info(
        "request_id=%s formatted_prompt=%s input_tokens=%d max_tokens=%d max_model_len=%d",
        request_id,
        prompt,
        len(prompt_token_ids),
        max_tokens,
        max_model_len,
    )
    if len(prompt_token_ids) >= max_model_len:
        raise HTTPException(
            status_code=400,
            detail=f"formatted prompt has {len(prompt_token_ids)} tokens; max_model_len={max_model_len} leaves no room for generation",
        )

    try:
        result = await request.app.state.engine_client.generate_with_details(
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            request_id=request_id,
        )
    except Exception:
        logger.exception("request_id=%s generation failed", request_id)
        raise

    generated_text = result["text"]
    prompt_tokens = result["prompt_tokens"]
    completion_tokens = result["completion_tokens"]
    total_tokens = result["total_tokens"]
    finish_reason = result["finish_reason"]
    logger.info(
        "request_id=%s final_text=%s prompt_tokens=%d output_tokens=%d total_tokens=%d finish_reason=%s termination_cause=%s max_tokens=%d max_model_len=%d",
        request_id,
        generated_text,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        finish_reason,
        result["termination_cause"],
        result["max_tokens"],
        result["max_model_len"],
    )

    response = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": generated_text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }
    return JSONResponse(content=response)


def attach_router(app: FastAPI):
    app.include_router(router)
