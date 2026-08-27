# ABOUTME: Covers the Analyst role for Case 3 without live Bedrock calls.
# ABOUTME: Verifies structured assessment output and stateless revision calls.
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import ValidationError

from restorebench.agents import analyst
from restorebench.llm.providers import ChatMessage, LLMResponse, ToolUse
from restorebench.schemas.actions import Maneuver
from restorebench.schemas.config import LLMAssignment, OrchestratorConfig
from restorebench.schemas.feedback import FailureFeedback
from restorebench.schemas.multi_agent import AnalystAssessment
from restorebench.schemas.power_flow import NRDiagnostics


MODEL_ID = "test-model"



def _assessment_payload(gen_id: int = 11) -> dict[str, Any]:
    return {
        "diagnosed_cause": "REACTIVE_DEFICIT",
        "proposed_maneuver": _maneuver_payload(gen_id),
        "rationale": "low voltage and Q limits suggest reactive support",
    }


def _run_analyst(
    monkeypatch: pytest.MonkeyPatch,
    responses: Sequence[LLMResponse],
    *,
    revision_guidance: str | None = None,
    history: Sequence[Maneuver] = (),
    failures: Sequence[FailureFeedback] = (),
    config: OrchestratorConfig | None = None,
) -> tuple[Any, ScriptedLLM]:
    fake_llm = ScriptedLLM(responses)
    monkeypatch.setattr(analyst, "llm_call", fake_llm)
    result = analyst.make_analyst(MODEL_ID)(
        card="## Scenario Card\nBus 44 is connected to bus 45.",
        diagnostics=_diagnostics(),
        history=history,
        failures=failures,
        config=config or _config(),
        revision_guidance=revision_guidance,
    )
    return result, fake_llm


def _maneuver_payload(gen_id: int = 11) -> dict[str, Any]:
    return {
        "action": {"type": "GEN_V_SETPOINT", "gen_id": gen_id, "new_vm_pu": 1.05},
        "diagnosed_cause": "REACTIVE_DEFICIT",
        "rationale": f"raise generator {gen_id} voltage",
    }


def _llm_response(tool_input: dict[str, Any] | None, *, text: str = "tool call") -> LLMResponse:
    tool_use = None
    assistant_content: tuple[dict[str, Any], ...] = ({"text": text},)
    if tool_input is not None:
        tool_use = ToolUse(toolUseId="toolu_assessment", name=analyst.ANALYST_TOOL_NAME, input=tool_input)
        assistant_content = ({"toolUse": tool_use.model_dump(mode="json", by_alias=True)},)
    return LLMResponse(
        text=text,
        model_id=MODEL_ID,
        tokens_in=101,
        tokens_out=202,
        latency_seconds=0.01,
        raw={"reasoning": "diagnose the weak pocket"},
        tool_use=tool_use,
        assistant_content=assistant_content,
    )


def _diagnostics() -> NRDiagnostics:
    return NRDiagnostics(
        iterations_attempted=30,
        worst_bus=44,
        lowest_vm_pu=0.71,
        lowest_vm_bus=45,
        gens_at_q_limit=[1, 2],
        max_mismatch_mw=12.5,
        max_mismatch_mvar=33.25,
        overstress=1.8,
        error_message="Newton-Raphson failed after 30 iterations",
        diagnostics_source="local_nose",
    )


def _config(configuration: int = 3) -> OrchestratorConfig:
    return OrchestratorConfig(
        CONFIGURATION=configuration,
        MANEUVER_BUDGET=10,
        MAX_RUNTIME_SECONDS=120,
        LLM_ASSIGNMENT=LLMAssignment(single_agent=None, analyst=MODEL_ID, executor=MODEL_ID, orchestrator=MODEL_ID),
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


def _message_text(messages: list[ChatMessage]) -> str:
    rendered: list[str] = []
    for message in messages:
        if isinstance(message.content, str):
            rendered.append(message.content)
        else:
            rendered.extend(str(block) for block in message.content)
    return "\n".join(rendered)


def test_analyst_happy_path_validates_assessment_and_tags_role(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _llm_response(_assessment_payload())

    result, _fake_llm = _run_analyst(monkeypatch, [response])

    assert result.assessment == AnalystAssessment.model_validate(_assessment_payload())
    assert result.llm_responses == (response,)
    assert response.raw["role"] == "analyst"


def test_analyst_request_uses_terminal_schema_thinking_and_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _llm_response(_assessment_payload())

    _result, fake_llm = _run_analyst(monkeypatch, [response])

    request = fake_llm.requests[0]
    tool_specs = {tool["toolSpec"]["name"]: tool["toolSpec"] for tool in request["tools"]["tools"]}
    assert request["thinking"] is True
    assert request["temperature"] == 1.0
    assert tool_specs[analyst.ANALYST_TOOL_NAME]["inputSchema"]["json"] == AnalystAssessment.model_json_schema()


def test_analyst_reprompts_once_with_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    malformed = _llm_response(None, text="plain text")
    valid = _llm_response(_assessment_payload())

    result, fake_llm = _run_analyst(monkeypatch, [malformed, valid])

    assert result.assessment == AnalystAssessment.model_validate(_assessment_payload())
    assert result.llm_responses == (malformed, valid)
    assert len(fake_llm.requests) == 2
    assert fake_llm.requests[1]["messages"][-2].content == list(malformed.assistant_content)
    assert "diagnosed_cause" in _message_text(fake_llm.requests[1]["messages"][-1:])
    assert "Field required" in _message_text(fake_llm.requests[1]["messages"][-1:])


def test_analyst_double_malformed_raises_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        _run_analyst(
            monkeypatch, [_llm_response(None, text="plain"), _llm_response({"diagnosed_cause": "BAD_SETPOINTS"})]
        )


def test_analyst_revision_call_is_fresh_and_contains_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llm = ScriptedLLM([_llm_response(_assessment_payload()), _llm_response(_assessment_payload(gen_id=12))])
    monkeypatch.setattr(analyst, "llm_call", fake_llm)
    role = analyst.make_analyst(MODEL_ID)

    role(
        card="card",
        diagnostics=_diagnostics(),
        history=(),
        failures=(),
        config=_config(),
    )
    role(
        card="card",
        diagnostics=_diagnostics(),
        history=(),
        failures=(),
        config=_config(),
        revision_guidance="Avoid generator 11; use downstream support.",
    )

    first_text = _message_text(fake_llm.requests[0]["messages"])
    second_text = _message_text(fake_llm.requests[1]["messages"])
    assert len(fake_llm.requests[1]["messages"]) == 2
    assert "Avoid generator 11; use downstream support." not in first_text
    assert "Avoid generator 11; use downstream support." in second_text
    assert "tool call" not in second_text
