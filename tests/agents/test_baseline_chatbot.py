# ABOUTME: Covers the Case-1 baseline chatbot proposer without live Bedrock calls.
# ABOUTME: Verifies one-shot sequence prompting, validation, re-prompting, and role tagging.
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import ValidationError

from restorebench.agents import baseline_chatbot
from restorebench.environment.orchestrator import ProposeSequence, ProposeSequenceResult
from restorebench.llm.providers import ChatMessage, LLMResponse, ToolUse
from restorebench.schemas.actions import ManeuverSequence
from restorebench.schemas.config import LLMAssignment, OrchestratorConfig


MODEL_ID = "test-model"


def _maneuver_payload(gen_id: int = 11) -> dict[str, Any]:
    return {
        "action": {"type": "GEN_V_SETPOINT", "gen_id": gen_id, "new_vm_pu": 1.05},
        "diagnosed_cause": "REACTIVE_DEFICIT",
        "rationale": f"raise generator {gen_id} voltage",
    }


def _sequence_payload(n_maneuvers: int = 1) -> dict[str, Any]:
    return {
        "maneuvers": [_maneuver_payload(gen_id=11 + index) for index in range(n_maneuvers)],
        "reconstruction_summary": "Buses around the weak pocket are connected through the listed corridors.",
    }


def _llm_response(
    tool_input: dict[str, Any] | None = None,
    *,
    raw: dict[str, Any] | None = None,
    text: str = "tool call",
    tool_name: str = baseline_chatbot.PROPOSE_SEQUENCE_TOOL_NAME,
) -> LLMResponse:
    tool_use = None
    assistant_content: tuple[dict[str, Any], ...] = ({"text": text},)
    if tool_input is not None:
        tool_use = ToolUse(toolUseId="toolu_sequence", name=tool_name, input=tool_input)
        assistant_content = ({"toolUse": tool_use.model_dump(mode="json", by_alias=True)},)
    return LLMResponse(
        text=text,
        model_id=MODEL_ID,
        tokens_in=101,
        tokens_out=202,
        latency_seconds=0.01,
        raw=raw or {},
        tool_use=tool_use,
        assistant_content=assistant_content,
    )


def _config(budget: int = 10) -> OrchestratorConfig:
    return OrchestratorConfig(
        CONFIGURATION=1,
        MANEUVER_BUDGET=budget,
        MAX_RUNTIME_SECONDS=120,
        LLM_ASSIGNMENT=LLMAssignment(single_agent=MODEL_ID, analyst=None, executor=None, orchestrator=None),
    )


class ScriptedLLM:
    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(
        self,
        model_id: str,
        messages: list[ChatMessage],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        thinking: bool = False,
        tools: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.requests.append(
            {
                "model_id": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "thinking": thinking,
                "tools": tools,
            }
        )
        return self.responses.pop(0)


def _run_chatbot(
    monkeypatch: pytest.MonkeyPatch,
    responses: Sequence[LLMResponse],
    *,
    budget: int = 10,
) -> tuple[ProposeSequenceResult, ScriptedLLM]:
    fake_llm = ScriptedLLM(responses)
    monkeypatch.setattr(baseline_chatbot, "llm_call", fake_llm)
    result = baseline_chatbot.make_baseline_chatbot(MODEL_ID)(
        card="## Scenario Card\nBus 44 is connected to bus 45.",
        config=_config(budget=budget),
    )
    return result, fake_llm


def _message_text(messages: list[ChatMessage]) -> str:
    rendered: list[str] = []
    for message in messages:
        if isinstance(message.content, str):
            rendered.append(message.content)
        else:
            rendered.extend(str(block) for block in message.content)
    return "\n".join(rendered)


def test_make_baseline_chatbot_returns_propose_sequence_protocol_callable() -> None:
    proposer: ProposeSequence = baseline_chatbot.make_baseline_chatbot(MODEL_ID)

    assert callable(proposer)


def test_happy_path_returns_validated_sequence_and_tags_reasoning_role(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _llm_response(_sequence_payload(), raw={"reasoning": "build graph then try voltage support"})

    result, _fake_llm = _run_chatbot(monkeypatch, [response])

    assert isinstance(result, ProposeSequenceResult)
    assert result.maneuvers == ManeuverSequence.model_validate(_sequence_payload())
    assert result.llm_responses == (response,)
    assert response.raw["role"] == "chatbot"


def test_request_shape_uses_thinking_temperature_and_generated_sequence_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _llm_response(_sequence_payload())

    _result, fake_llm = _run_chatbot(monkeypatch, [response])

    request = fake_llm.requests[0]
    tool_specs = {tool["toolSpec"]["name"]: tool["toolSpec"] for tool in request["tools"]["tools"]}
    assert request["model_id"] == MODEL_ID
    assert request["thinking"] is True
    assert request["temperature"] == 1.0
    assert set(tool_specs) == {baseline_chatbot.PROPOSE_SEQUENCE_TOOL_NAME}
    assert tool_specs[baseline_chatbot.PROPOSE_SEQUENCE_TOOL_NAME]["inputSchema"]["json"] == (
        ManeuverSequence.model_json_schema()
    )


def test_prompt_contains_case1_context_but_no_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _llm_response(_sequence_payload())

    _result, fake_llm = _run_chatbot(monkeypatch, [response], budget=3)

    messages = fake_llm.requests[0]["messages"]
    system_text = messages[0].content
    user_text = messages[1].content
    full_text = _message_text(messages)
    assert "## Scenario Card\nBus 44 is connected to bus 45." in user_text
    assert "AC power flow does not converge" in user_text
    assert "reconstruct" in system_text.lower()
    assert "reconstruction_summary" in system_text
    assert "applied in order" in system_text
    assert "stopping at first convergence" in system_text
    assert "no feedback between maneuvers" in system_text
    assert "MANEUVER_BUDGET=3" in system_text
    assert "# Latest NR diagnostics" not in full_text
    assert "worst_bus" not in full_text
    assert "gens_at_q_limit" not in full_text
    assert "# Accepted maneuver history" not in full_text
    assert "# Failure feedback so far" not in full_text


def test_malformed_first_response_reprompts_once_with_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = _llm_response(None, text="I would raise a generator.")
    valid = _llm_response(_sequence_payload())

    result, fake_llm = _run_chatbot(monkeypatch, [malformed, valid])

    assert result.maneuvers == ManeuverSequence.model_validate(_sequence_payload())
    assert result.llm_responses == (malformed, valid)
    assert len(fake_llm.requests) == 2
    second_messages = fake_llm.requests[1]["messages"]
    assert second_messages[-2].role == "assistant"
    assert second_messages[-2].content == list(malformed.assistant_content)
    assert second_messages[-1].role == "user"
    assert "maneuvers" in _message_text(second_messages[-1:])
    assert "Field required" in _message_text(second_messages[-1:])
    assert malformed.raw["role"] == "chatbot"
    assert valid.raw["role"] == "chatbot"


def test_double_malformed_response_raises_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    malformed = [_llm_response(None, text="not structured"), _llm_response({"maneuvers": []})]

    with pytest.raises(ValidationError):
        _run_chatbot(monkeypatch, malformed)


def test_baseline_chatbot_imports_no_backend_tools() -> None:
    source_path = baseline_chatbot.__file__
    assert source_path is not None
    with open(source_path, encoding="utf-8") as handle:
        assert "restorebench.tools" not in handle.read()
