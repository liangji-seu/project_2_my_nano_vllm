# Pure helpers for structured generation completion metadata.


def get_termination_cause(
    finish_reason: str,
    prompt_tokens: int,
    output_tokens: int,
    max_tokens: int,
    max_model_len: int,
    *,
    eos_reached: bool = False,
) -> str:
    if finish_reason == "stop":
        return "eos" if eos_reached else "stop"
    if finish_reason == "length":
        hit_max_tokens = output_tokens >= max_tokens
        hit_context = prompt_tokens + output_tokens >= max_model_len
        if hit_max_tokens and hit_context:
            return "max_tokens_and_context_limit"
        if hit_max_tokens:
            return "max_tokens"
        if hit_context:
            return "context_limit"
        return "length_limit"
    return {"abort": "aborted", "error": "error"}.get(finish_reason, "unknown")


def should_log_stream_update(
    output_count: int,
    last_logged_count: int,
    last_logged_at: float,
    now: float,
    finished: bool,
    *,
    every_tokens: int,
    interval_seconds: float,
) -> bool:
    if output_count <= last_logged_count:
        return False
    return (
        output_count == 1
        or output_count - last_logged_count >= every_tokens
        or now - last_logged_at >= interval_seconds
        or finished
    )
