import asyncio
import json
import queue
from types import SimpleNamespace

import pytest

from my_vllm.engine.async_llm import AsyncLLM
from my_vllm.engine.core import EngineCoreProc
from my_vllm.engine.core_client import MPClient
from my_vllm.engine.output_details import (
    get_termination_cause,
    should_log_stream_update,
)
from my_vllm.routes.chat import chat_completions, prepare_chat_prompt


class FakeTokenizer:
    chat_template = "local test template"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<|im_start|>{messages[0]['role']}\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(range(31)) if "<|im_start|>" in text else list(range(len(text.split())))


class QueueOutputSocket:
    def __init__(self):
        self.items = asyncio.Queue()

    async def recv_json(self):
        return await self.items.get()


class RoutingInputSocket:
    def __init__(self, output_socket, result):
        self.output_socket = output_socket
        self.result = result
        self.requests = []

    async def send_multipart(self, frames):
        request = json.loads(frames[1])
        self.requests.append(request)
        await self.output_socket.items.put(
            {**self.result, "request_id": request["request_id"]}
        )


class FakeRequest:
    def __init__(self, state, body):
        self.app = SimpleNamespace(state=state)
        self.body = body

    async def json(self):
        return self.body


def test_template_prompt_and_raw_content_token_counts_are_distinct():
    prompt, token_ids, raw_tokens = prepare_chat_prompt(
        [{"role": "user", "content": "介绍 阿里巴巴"}], FakeTokenizer()
    )
    assert prompt.startswith("<|im_start|>user\n")
    assert prompt.endswith("<|im_start|>assistant\n")
    assert len(token_ids) == 31
    assert raw_tokens == 2


def test_route_async_client_transport_and_legacy_str_result():
    async def scenario():
        result = {
            "text": "测" * 238,
            "prompt_tokens": 31,
            "completion_tokens": 128,
            "total_tokens": 159,
            "finish_reason": "length",
            "termination_cause": "max_tokens",
            "max_tokens": 128,
            "max_model_len": 1024,
        }
        output_socket = QueueOutputSocket()
        input_socket = RoutingInputSocket(output_socket, result)
        client = object.__new__(MPClient)
        client._started = True
        client._engine_identities = [b"engine"]
        client._input_socket = input_socket
        client._output_socket = output_socket
        client._pending_requests = {}
        output_task = asyncio.create_task(MPClient._process_outputs(client))

        llm = object.__new__(AsyncLLM)
        llm._started = True
        llm.mp_client = client
        state = SimpleNamespace(
            chat_tokenizer=FakeTokenizer(),
            served_model_name="qwen-test",
            vllm_config=SimpleNamespace(max_model_len=1024),
            engine_client=llm,
        )
        body = {
            "messages": [{"role": "user", "content": "介绍 阿里巴巴"}],
            "max_tokens": 128,
            # Untrusted client fields are ignored; IDs come only from local template tokenizer.
            "prompt_token_ids": [999],
        }
        response = await chat_completions(FakeRequest(state, body))
        payload = json.loads(response.body)
        sent = input_socket.requests[0]

        assert sent["prompt"].startswith("<|im_start|>user\n")
        assert sent["prompt"].endswith("<|im_start|>assistant\n")
        assert sent["prompt_token_ids"] != [999]
        assert payload["usage"] == {
            "prompt_tokens": 31,
            "completion_tokens": 128,
            "total_tokens": 159,
        }
        assert payload["choices"][0]["finish_reason"] == "length"
        assert len(payload["choices"][0]["message"]["content"]) == 238

        assert await AsyncLLM.generate(llm, "legacy prompt") == result["text"]
        assert len(input_socket.requests) == 2
        output_task.cancel()
        await output_task
        assert output_task.done()


    asyncio.run(scenario())

def test_core_build_request_uses_internal_formatted_token_ids():
    class Tokenizer:
        eos_token_ids = (2,)

        def encode(self, _text):
            raise AssertionError("must not re-encode supplied IDs")

    core = object.__new__(EngineCoreProc)
    core.tokenizer = Tokenizer()
    core.vllm_config = SimpleNamespace(max_model_len=1024)
    request = EngineCoreProc._build_request(
        core,
        {
            "request_id": "req1",
            "prompt": "formatted prompt",
            "prompt_token_ids": [10, 11, 12],
            "max_tokens": 5,
        },
    )
    assert request.prompt_token_ids == [10, 11, 12]
    assert request.num_prompt_tokens == 3


def test_core_completion_payload_has_true_counts_and_reason(caplog):
    class Request:
        output_token_ids = [101, 102]
        num_prompt_tokens = 16
        num_output_tokens = 128
        num_tokens = 144
        max_tokens = 128

        @staticmethod
        def get_finished_reason():
            return SimpleNamespace(value="length")

    class Tokenizer:
        eos_token_ids = (2,)

        @staticmethod
        def decode(token_ids):
            return "output text"

    request = Request()
    core = object.__new__(EngineCoreProc)
    core.tokenizer = Tokenizer()
    core.vllm_config = SimpleNamespace(max_model_len=1024)
    core.scheduler = SimpleNamespace(
        finished_req_ids={"req1"}, requests={"req1": request}
    )
    core.output_queue = queue.Queue()
    core._streamed_text = {"req1": (1, "prefix", 0.0)}
    with caplog.at_level("INFO", logger="my_vllm.engine.core"):
        EngineCoreProc._send_finished_outputs(core)
    output = core.output_queue.get_nowait()
    assert any("final_flush=True" in record.message for record in caplog.records)

    assert output == {
        "request_id": "req1",
        "text": "output text",
        "prompt_tokens": 16,
        "completion_tokens": 128,
        "total_tokens": 144,
        "finish_reason": "length",
        "max_tokens": 128,
        "max_model_len": 1024,
        "termination_cause": "max_tokens",
    }


@pytest.mark.parametrize(
    ("reason", "prompt", "output", "max_tokens", "max_len", "eos", "expected"),
    [
        ("stop", 5, 1, 10, 100, True, "eos"),
        ("stop", 5, 1, 10, 100, False, "stop"),
        ("length", 16, 128, 128, 1024, False, "max_tokens"),
        ("length", 100, 20, 128, 120, False, "context_limit"),
        ("length", 10, 10, 10, 20, False, "max_tokens_and_context_limit"),
    ],
)
def test_termination_cause_boundaries(reason, prompt, output, max_tokens, max_len, eos, expected):
    assert get_termination_cause(
        reason, prompt, output, max_tokens, max_len, eos_reached=eos
    ) == expected


def test_stream_logging_thresholds():
    should_log = should_log_stream_update
    assert should_log(1, 0, 0.0, 0.01, False, every_tokens=16, interval_seconds=0.25)
    assert not should_log(15, 1, 0.01, 0.20, False, every_tokens=16, interval_seconds=0.25)
    assert should_log(17, 1, 0.01, 0.20, False, every_tokens=16, interval_seconds=0.25)
    assert should_log(2, 1, 0.01, 0.26, False, every_tokens=16, interval_seconds=0.25)
    assert should_log(2, 1, 0.20, 0.21, True, every_tokens=16, interval_seconds=0.25)
    assert not should_log(1, 1, 0.01, 2.0, True, every_tokens=16, interval_seconds=0.25)
