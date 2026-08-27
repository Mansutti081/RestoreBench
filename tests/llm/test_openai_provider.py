# ABOUTME: Covers the OpenAI Responses-API transport: input/tool translation both ways,
# ABOUTME: reasoning fields, truncation and refusal handling, and the output-item echo protocol.
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from restorebench.llm import models, openai_provider
from restorebench.llm.providers import ChatMessage, llm_call
from restorebench.schemas.errors import LLMFailureError


class _Item:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def model_dump(self, mode="json", exclude_none=True):
        return dict(self._payload)


def _message_item(text: str):
    return _Item({"type": "message", "role": "assistant",
                  "content": [{"type": "output_text", "text": text}]})


def _response(*, output=None, status="completed", incomplete_reason=None):
    return SimpleNamespace(
        output=output if output is not None else [_message_item("answer")],
        status=status,
        incomplete_details=SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None,
        usage=SimpleNamespace(input_tokens=100, output_tokens=40, total_tokens=150),
    )


class _FakeClient:
    def __init__(self, response):
        self.requests: list[dict[str, Any]] = []
        outer = self

        class _Responses:
            def create(self, **kwargs):
                outer.requests.append(kwargs)
                return response

        self.responses = _Responses()


@pytest.fixture
def fake(monkeypatch):
    def install(response):
        client = _FakeClient(response)
        monkeypatch.setattr(openai_provider, "_client", lambda: client)
        return client

    return install


def test_dispatch_routes_the_openai_id_through_this_transport(fake):
    client = fake(_response())

    response = llm_call(models.OPENAI_SOL, [ChatMessage(role="user", content="hi")])

    assert client.requests[0]["model"] == models.OPENAI_SOL
    assert response.text == "answer"
    assert response.tokens_in == 100 and response.tokens_out == 40 and response.tokens_total == 150


def test_system_lifts_into_instructions_reasoning_set_and_no_temperature(fake):
    client = fake(_response())

    llm_call(
        models.OPENAI_SOL,
        [ChatMessage(role="system", content="rules"), ChatMessage(role="user", content="go")],
        temperature=0.7,
        thinking=True,
    )

    request = client.requests[0]
    assert request["instructions"] == "rules"
    assert request["input"] == [{"role": "user", "content": "go"}]
    assert request["reasoning"] == {"effort": "high"}
    assert "temperature" not in request


def test_converse_tool_config_translates_to_flat_function_specs(fake):
    client = fake(_response())
    tools = {"tools": [
        {"toolSpec": {"name": "probe", "description": "d", "inputSchema": {"json": {"type": "object"}}}}
    ]}

    llm_call(models.OPENAI_SOL, [ChatMessage(role="user", content="x")], tools=tools)

    [spec] = client.requests[0]["tools"]
    assert spec == {"type": "function", "name": "probe", "description": "d",
                    "parameters": {"type": "object"}}


def test_function_calls_parse_and_echo_keeps_reasoning_items(fake):
    fake(_response(output=[
        _Item({"type": "reasoning", "id": "rs_1", "summary": []}),
        _Item({"type": "function_call", "call_id": "call_1", "name": "probe",
               "arguments": '{"gen_id": 3}'}),
    ]))

    response = llm_call(models.OPENAI_SOL, [ChatMessage(role="user", content="x")])

    [use] = response.tool_uses
    assert use.tool_use_id == "call_1" and use.name == "probe" and use.input == {"gen_id": 3}
    # the echo carries the reasoning item: the API rejects a function_call without it next turn
    assert response.assistant_content[0]["type"] == "reasoning"
    assert response.assistant_content[1]["type"] == "function_call"


def test_echoed_turn_and_tool_result_become_input_items(fake):
    client = fake(_response())
    messages = [
        ChatMessage(role="user", content="start"),
        ChatMessage(role="assistant", content=[
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "function_call", "call_id": "call_1", "name": "probe",
             "arguments": '{"gen_id": 3}'},
        ]),
        ChatMessage(role="user", content=[
            {"toolResult": {"toolUseId": "call_1", "content": [{"json": {"ok": True}}],
                            "status": "success"}},
            {"text": "now propose"},
        ]),
    ]

    llm_call(models.OPENAI_SOL, messages)

    items = client.requests[0]["input"]
    assert items[0] == {"role": "user", "content": "start"}
    assert items[1]["type"] == "reasoning"
    assert items[2]["type"] == "function_call"
    assert items[3] == {"type": "function_call_output", "call_id": "call_1",
                        "output": '{"ok": true}'}
    assert items[4] == {"role": "user", "content": "now propose"}


def test_max_output_tokens_truncation_raises_unless_opted_out(fake):
    fake(_response(status="incomplete", incomplete_reason="max_output_tokens"))
    with pytest.raises(LLMFailureError, match="truncated"):
        llm_call(models.OPENAI_SOL, [ChatMessage(role="user", content="x")])

    fake(_response(status="incomplete", incomplete_reason="max_output_tokens"))
    response = llm_call(models.OPENAI_SOL, [ChatMessage(role="user", content="x")],
                        raise_on_truncation=False)
    assert response.raw["status"] == "incomplete"


def test_refusal_raises_the_shared_failure_type(fake):
    fake(_response(output=[_Item({"type": "message", "role": "assistant",
                                  "content": [{"type": "refusal", "refusal": "cannot help"}]})]))

    with pytest.raises(LLMFailureError, match="declined"):
        llm_call(models.OPENAI_SOL, [ChatMessage(role="user", content="x")])


def test_malformed_function_arguments_raise_the_shared_failure_type(fake):
    fake(_response(output=[_Item({"type": "function_call", "call_id": "c", "name": "probe",
                                  "arguments": "{not json"})]))

    with pytest.raises(LLMFailureError, match="not valid JSON"):
        llm_call(models.OPENAI_SOL, [ChatMessage(role="user", content="x")])
