# ABOUTME: Covers the uniform LLM provider abstraction without live Bedrock calls.
# ABOUTME: Mocks the boto3 client boundary so no AWS credentials are required.
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest
from botocore.exceptions import ClientError
from pydantic import ValidationError

from restorebench.llm import providers
from restorebench.llm.models import DEEPSEEK_V3_2, GLM_5, GPT_OSS_120B, HAIKU_4_5, KIMI_K2_5, OPUS_4_6, QWEN3_32B
from restorebench.llm.providers import ChatMessage, LLMResponse, ToolResult, llm_call
from restorebench.schemas.errors import LLMFailureError

OPUS = OPUS_4_6
HAIKU = HAIKU_4_5
QWEN = QWEN3_32B
GPT_OSS = GPT_OSS_120B
KIMI = KIMI_K2_5
GLM5 = GLM_5
DEEPSEEK = DEEPSEEK_V3_2


def _client_error(code: str, status: int) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        "Converse",
    )


def _converse_result(
    text: str = "ok",
    *,
    reasoning: bool = False,
    tool_use: dict | None = None,
    tool_uses: list[dict] | None = None,
) -> dict:
    blocks: list[dict] = []
    if reasoning:
        blocks.append({"reasoningContent": {"reasoningText": {"text": "let me think"}}})
    if tool_use is not None:
        blocks.append({"toolUse": tool_use})
    for extra_tool_use in tool_uses or []:
        blocks.append({"toolUse": extra_tool_use})
    blocks.append({"text": text})
    return {
        "output": {"message": {"content": blocks}},
        "usage": {"inputTokens": 11, "outputTokens": 7},
        "stopReason": "tool_use" if tool_use is not None or tool_uses else "end_turn",
    }


def _mock_client(monkeypatch, result=None, side_effect=None) -> Mock:
    client = Mock()
    if side_effect is not None:
        client.converse.side_effect = side_effect
    else:
        client.converse.return_value = result if result is not None else _converse_result()
    monkeypatch.setattr(providers, "_bedrock_client", lambda: client)
    return client


def test_bedrock_client_is_cached_after_first_creation(monkeypatch):
    created = []

    def fake_client(service_name, *, region_name, config):
        client = SimpleNamespace(service_name=service_name, region_name=region_name, config=config, index=len(created))
        created.append(client)
        return client

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=fake_client))
    monkeypatch.setattr(providers, "_BEDROCK_CLIENT", None, raising=False)
    monkeypatch.setenv("AWS_REGION", "eu-central-1")

    first = providers._bedrock_client()
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    second = providers._bedrock_client()

    assert first is second
    assert len(created) == 1
    assert first.service_name == "bedrock-runtime"
    assert first.region_name == "eu-central-1"


def test_chat_message_and_response_are_strict_models():
    message = ChatMessage(role="user", content="Diagnose this grid.")
    response = LLMResponse(text="ok", model_id=OPUS, tokens_in=3, tokens_out=1, latency_seconds=0.1)

    assert message.role == "user"
    assert response.raw == {}
    with pytest.raises(ValidationError):
        ChatMessage(role="tool", content="bad")
    with pytest.raises(ValidationError):
        LLMResponse(text="ok", model_id=OPUS, tokens_in=3, tokens_out=1, latency_seconds=0.1, extra="forbidden")


def test_llm_failure_error_carries_model_id_and_underlying_exception():
    exc = LLMFailureError(model_id=OPUS, underlying_exception="timeout")

    assert exc.model_id == OPUS
    assert exc.underlying_exception == "timeout"
    assert OPUS in str(exc)
    assert "timeout" in str(exc)


def test_llm_call_rejects_empty_messages_before_provider_dispatch():
    with pytest.raises(LLMFailureError) as error:
        llm_call(OPUS, [])

    assert error.value.model_id == OPUS
    assert "at least one message" in error.value.underlying_exception


def test_converse_request_splits_system_prompt_and_carries_inference_config(monkeypatch):
    client = _mock_client(monkeypatch)
    messages = [
        ChatMessage(role="system", content="You are a grid operator."),
        ChatMessage(role="user", content="Bus 3 diverges."),
    ]

    llm_call(OPUS, messages, temperature=1.0, max_tokens=256)

    kwargs = client.converse.call_args.kwargs
    assert kwargs["modelId"] == OPUS
    assert kwargs["system"] == [{"text": "You are a grid operator."}]
    assert kwargs["messages"] == [{"role": "user", "content": [{"text": "Bus 3 diverges."}]}]
    assert kwargs["inferenceConfig"] == {"maxTokens": 256, "temperature": 1.0}
    assert "additionalModelRequestFields" not in kwargs


def test_converse_request_forwards_tool_config(monkeypatch):
    client = _mock_client(monkeypatch)
    tools = {
        "tools": [
            {
                "toolSpec": {
                    "name": "propose_maneuvers",
                    "description": "Return a maneuver sequence.",
                    "inputSchema": {"json": {"type": "object"}},
                }
            }
        ]
    }

    llm_call(OPUS, [ChatMessage(role="user", content="fix it")], tools=tools)

    assert client.converse.call_args.kwargs["toolConfig"] == tools


def test_converse_response_maps_text_tokens_and_stop_reason(monkeypatch):
    _mock_client(monkeypatch, result=_converse_result("raise shunt"))

    response = llm_call(QWEN, [ChatMessage(role="user", content="fix it")])

    assert response.text == "raise shunt"
    assert response.model_id == QWEN
    assert response.tokens_in == 11
    assert response.tokens_out == 7
    assert response.latency_seconds >= 0
    assert response.raw["stop_reason"] == "end_turn"


def test_tool_use_response_exposes_validated_tool_input_unmodified(monkeypatch):
    tool_input = {
        "maneuvers": [
            {
                "action": {"type": "GEN_V_SETPOINT", "gen_id": 11, "new_vm_pu": 1.05},
                "diagnosed_cause": "REACTIVE_DEFICIT",
                "rationale": "raise local voltage support",
            }
        ],
        "reconstruction_summary": "read the card",
    }
    tool_use = {"toolUseId": "toolu_1", "name": "propose_maneuvers", "input": tool_input}
    _mock_client(monkeypatch, result=_converse_result(tool_use=tool_use, reasoning=True))

    response = llm_call(QWEN, [ChatMessage(role="user", content="fix it")], thinking=True)

    assert response.raw["stop_reason"] == "tool_use"
    assert response.tool_use is not None
    assert response.tool_use.tool_use_id == "toolu_1"
    assert response.tool_use.name == "propose_maneuvers"
    assert response.tool_use.input == tool_input
    assert response.tool_uses == (response.tool_use,)


def test_anthropic_tool_use_extra_type_key_is_tolerated(monkeypatch):
    tool_input = {
        "maneuvers": [
            {
                "action": {"type": "TAP_ADJUSTMENT", "trafo_id": 2, "new_tap_pos": -2},
                "diagnosed_cause": "BAD_SETPOINTS",
                "rationale": "adjust transformer ratio",
            }
        ],
        "reconstruction_summary": "read the card",
    }
    tool_use = {
        "toolUseId": "toolu_anthropic",
        "name": "propose_maneuvers",
        "input": tool_input,
        "type": "tool_use",
    }
    _mock_client(monkeypatch, result=_converse_result(tool_use=tool_use, reasoning=True))

    response = llm_call(OPUS, [ChatMessage(role="user", content="fix it")], thinking=True)

    assert response.tool_use is not None
    assert response.tool_use.input == tool_input


def test_multiple_tool_uses_and_assistant_blocks_are_exposed(monkeypatch):
    first = {"toolUseId": "toolu_1", "name": "get_grid_topology", "input": {}}
    second = {
        "toolUseId": "toolu_2",
        "name": "get_action_applicability",
        "input": {"action": {"type": "GEN_V_SETPOINT", "gen_id": 3, "new_vm_pu": 1.02}},
    }
    _mock_client(monkeypatch, result=_converse_result(tool_uses=[first, second], reasoning=True))

    response = llm_call(OPUS, [ChatMessage(role="user", content="fix it")], thinking=True)

    assert [tool.name for tool in response.tool_uses] == ["get_grid_topology", "get_action_applicability"]
    assert response.tool_use == response.tool_uses[0]
    assert response.raw["tool_uses"][1]["toolUseId"] == "toolu_2"
    assert response.assistant_content == tuple(response.raw["assistant_content"])
    assert response.assistant_content[0] == {"reasoningContent": {"reasoningText": {"text": "let me think"}}}


def test_continuation_replays_assistant_blocks_and_sends_tool_results(monkeypatch):
    assistant_blocks = [
        {"reasoningContent": {"reasoningText": {"text": "inspect topology"}}},
        {"toolUse": {"toolUseId": "toolu_1", "name": "get_grid_topology", "input": {}}},
        {
            "toolUse": {
                "toolUseId": "toolu_2",
                "name": "get_action_applicability",
                "input": {"action": {"type": "SHUNT_STEP", "shunt_id": 0, "new_step": 0}},
            }
        },
    ]
    tool_result_blocks = [
        {
            "toolResult": ToolResult(
                toolUseId="toolu_1",
                content=[{"json": {"n_buses": 118}}],
                status="success",
            ).model_dump(mode="json", by_alias=True)
        },
        {
            "toolResult": ToolResult(
                toolUseId="toolu_2",
                content=[{"json": {"applicable": True, "reason": None}}],
                status="success",
            ).model_dump(mode="json", by_alias=True)
        },
    ]
    client = _mock_client(monkeypatch, result=_converse_result("continue without fresh reasoning", reasoning=False))

    response = llm_call(
        OPUS,
        [
            ChatMessage(role="system", content="You are a grid operator."),
            ChatMessage(role="user", content="fix it"),
            ChatMessage(role="assistant", content=assistant_blocks),
            ChatMessage(role="user", content=tool_result_blocks),
        ],
        thinking=True,
    )

    kwargs = client.converse.call_args.kwargs
    assert kwargs["messages"][1] == {"role": "assistant", "content": assistant_blocks}
    assert kwargs["messages"][2] == {"role": "user", "content": tool_result_blocks}
    assert response.text == "continue without fresh reasoning"
    assert "reasoning" not in response.raw


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        (OPUS, {"thinking": {"type": "adaptive"}}),
        (HAIKU, {"reasoning_config": {"type": "enabled", "budget_tokens": providers.THINKING_BUDGET_TOKENS}}),
        (DEEPSEEK, {"reasoning_effort": "high"}),
        (KIMI, {"reasoning_effort": "high"}),
        (GLM5, {"reasoning_effort": "high"}),
        (QWEN, {"reasoning_effort": "high"}),
        (GPT_OSS, {"reasoning_effort": "high"}),
    ],
)
def test_thinking_flag_sends_the_provider_specific_reasoning_field(monkeypatch, model_id, expected):
    client = _mock_client(monkeypatch, result=_converse_result(reasoning=True))

    llm_call(model_id, [ChatMessage(role="user", content="fix it")], thinking=True)

    assert client.converse.call_args.kwargs["additionalModelRequestFields"] == expected


def test_thinking_on_an_unknown_model_raises_instead_of_guessing_the_reasoning_field(monkeypatch):
    """Each provider names the reasoning field differently and Bedrock ignores an
    unrecognised one silently. Guessing a default for an unknown model would run the
    whole benchmark without reasoning and report nothing."""
    _mock_client(monkeypatch, result=_converse_result(reasoning=True))

    with pytest.raises(LLMFailureError) as error:
        llm_call("anthropic.claude-opus-4-8", [ChatMessage(role="user", content="fix it")], thinking=True)

    assert "unknown model" in error.value.underlying_exception.lower()


def test_reasoning_fields_are_keyed_on_exact_model_ids(monkeypatch):
    for model_id in (OPUS, HAIKU, QWEN, GPT_OSS, DEEPSEEK):
        assert providers._reasoning_fields(model_id)


def test_thinking_response_exposes_reasoning_text_in_raw(monkeypatch):
    _mock_client(monkeypatch, result=_converse_result("raise shunt", reasoning=True))

    response = llm_call(DEEPSEEK, [ChatMessage(role="user", content="fix it")], thinking=True)

    assert response.text == "raise shunt"
    assert response.raw["reasoning"] == "let me think"


def test_thinking_requested_but_absent_from_response_raises_instead_of_silently_degrading(monkeypatch):
    """Bedrock ignores unknown additionalModelRequestFields silently, so a typo would
    otherwise produce a full benchmark run at reduced capability with no warning."""
    _mock_client(monkeypatch, result=_converse_result("raise shunt", reasoning=False))

    with pytest.raises(LLMFailureError) as error:
        llm_call(GPT_OSS, [ChatMessage(role="user", content="fix it")], thinking=True)

    assert "reasoning" in error.value.underlying_exception.lower()


def test_llm_call_retries_throttling_then_succeeds(monkeypatch):
    monkeypatch.setattr(providers, "_sleep", lambda _seconds: None)
    client = _mock_client(
        monkeypatch,
        side_effect=[_client_error("ThrottlingException", 429), _converse_result("ok")],
    )

    response = llm_call(QWEN, [ChatMessage(role="user", content="fix it")])

    assert response.text == "ok"
    assert client.converse.call_count == 2


def test_llm_call_retries_transient_failures_then_raises_llm_failure(monkeypatch):
    monkeypatch.setattr(providers, "_sleep", lambda _seconds: None)
    client = _mock_client(monkeypatch, side_effect=_client_error("ServiceUnavailableException", 503))

    with pytest.raises(LLMFailureError) as error:
        llm_call(DEEPSEEK, [ChatMessage(role="user", content="fix it")])

    assert error.value.model_id == DEEPSEEK
    assert client.converse.call_count == providers.MAX_RETRIES + 1


def test_llm_call_does_not_retry_access_denied(monkeypatch):
    monkeypatch.setattr(providers, "_sleep", lambda _seconds: None)
    client = _mock_client(monkeypatch, side_effect=_client_error("AccessDeniedException", 403))

    with pytest.raises(LLMFailureError):
        llm_call(OPUS, [ChatMessage(role="user", content="fix it")])

    assert client.converse.call_count == 1


def test_llm_call_does_not_retry_validation_errors(monkeypatch):
    monkeypatch.setattr(providers, "_sleep", lambda _seconds: None)
    client = _mock_client(monkeypatch, side_effect=_client_error("ValidationException", 400))

    with pytest.raises(LLMFailureError):
        llm_call(GPT_OSS, [ChatMessage(role="user", content="fix it")])

    assert client.converse.call_count == 1


def test_adaptive_thinking_model_may_skip_reasoning_on_first_turn(monkeypatch):
    """Opus uses {'thinking': {'type': 'adaptive'}}: the model itself decides when to
    think, so a first-turn response without a reasoning block is a legitimate answer,
    not a mistyped-field failure. Raising here would record every such run as
    LLM_FAILURE and depress Opus's success rate artificially."""
    _mock_client(monkeypatch, result=_converse_result("raise shunt", reasoning=False))

    response = llm_call(OPUS, [ChatMessage(role="user", content="fix it")], thinking=True)

    assert response.text == "raise shunt"
    assert "reasoning" not in response.raw


def test_budget_thinking_model_without_reasoning_still_raises(monkeypatch):
    """Budget/effort-style reasoning fields guarantee emission; a missing block there
    still means the field was mistyped or silently ignored — the original guard stays."""
    _mock_client(monkeypatch, result=_converse_result("raise shunt", reasoning=False))

    with pytest.raises(LLMFailureError):
        llm_call(HAIKU, [ChatMessage(role="user", content="fix it")], thinking=True)


def test_bedrock_client_sets_explicit_timeouts(monkeypatch):
    """The deleted benchmark client pinned timeout=180; botocore's 60s default read
    timeout silently regressed long generations. The client must pin its own budget."""
    monkeypatch.setattr(providers, "_BEDROCK_CLIENT", None)
    captured = {}

    def fake_client(service_name, *, region_name, config):
        captured["config"] = config
        return SimpleNamespace(service_name=service_name, region_name=region_name)

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=fake_client))
    providers._bedrock_client()

    assert captured["config"].read_timeout == providers.LLM_READ_TIMEOUT_SECONDS == 180
    assert captured["config"].connect_timeout == 10


def test_response_captures_bedrock_total_tokens(monkeypatch):
    """Bedrock reports totalTokens itself; it can exceed inputTokens+outputTokens when
    cache-read/write tokens are billed, so record what Bedrock says rather than summing."""
    result = _converse_result("ok")
    result["usage"] = {"inputTokens": 11, "outputTokens": 7, "totalTokens": 25}
    _mock_client(monkeypatch, result=result)

    response = llm_call(GLM5, [ChatMessage(role="user", content="fix it")])

    assert response.tokens_in == 11
    assert response.tokens_out == 7
    assert response.tokens_total == 25


def test_total_tokens_falls_back_to_the_sum_when_bedrock_omits_it(monkeypatch):
    _mock_client(monkeypatch, result=_converse_result("ok"))  # usage has no totalTokens

    response = llm_call(GLM5, [ChatMessage(role="user", content="fix it")])

    assert response.tokens_total == 18  # 11 + 7


def test_a_response_truncated_at_max_tokens_raises_instead_of_looking_malformed(monkeypatch):
    """A max_tokens stop is our cap biting, not the model failing.

    Bedrock returns the partial message: the toolUse block is cut off or absent, so the caller
    sees an empty tool input and records MALFORMED_OUTPUT — charging a harness limit to the
    model. The reasoning budget is spent inside max_tokens, which makes this easy to hit.
    """
    truncated = _converse_result("half a thou")
    truncated["stopReason"] = "max_tokens"
    _mock_client(monkeypatch, result=truncated)

    with pytest.raises(LLMFailureError) as error:
        llm_call(HAIKU, [ChatMessage(role="user", content="fix it")])

    assert "max_tokens" in error.value.underlying_exception


def test_a_caller_can_opt_out_of_raising_on_truncation(monkeypatch):
    """The standalone benchmark harness records truncation as its own outcome, so it must be able
    to receive the truncated response instead of an exception. Backend agents keep the default raise.
    """
    truncated = _converse_result("half a thou")
    truncated["stopReason"] = "max_tokens"
    _mock_client(monkeypatch, result=truncated)

    response = llm_call(HAIKU, [ChatMessage(role="user", content="fix it")], raise_on_truncation=False)

    assert response.raw["stop_reason"] == "max_tokens"


def test_the_output_cap_clears_observed_opus_analyst_output():
    # Opus as analyst measured ~7000-8000 output tokens per call (one hit 7916); the cap must sit
    # well above that so verbose reasoning does not truncate the tool call.
    assert providers.DEFAULT_MAX_TOKENS >= 12000


def test_the_output_cap_leaves_room_for_the_reasoning_budget(monkeypatch):
    client = _mock_client(monkeypatch, result=_converse_result("ok", reasoning=True))

    llm_call(HAIKU, [ChatMessage(role="user", content="fix it")], thinking=True)

    max_tokens = client.converse.call_args.kwargs["inferenceConfig"]["maxTokens"]
    assert max_tokens > providers.THINKING_BUDGET_TOKENS * 2, (
        "reasoning tokens are billed inside maxTokens; too tight a cap truncates the tool call"
    )
