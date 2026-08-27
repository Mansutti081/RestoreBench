# ABOUTME: Covers the Executor role for multi-agent configurations without live Bedrock calls.
# ABOUTME: Verifies shared diagnostic tool loop use and deterministic report construction with real tools.
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandapower as pp
import pytest

from restorebench.agents import executor
from restorebench.llm.providers import ChatMessage, LLMResponse, ToolUse
from restorebench.schemas.actions import Maneuver
from restorebench.schemas.config import LLMAssignment, OrchestratorConfig
from restorebench.schemas.multi_agent import AnalystAssessment, ExecutorReport
from restorebench.tools.sandbox import create_sandbox, discard_sandbox, resolve_net


MODEL_ID = "test-model"


def _tiny_net() -> Any:
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=110.0, name="slack")
    b1 = pp.create_bus(net, vn_kv=110.0, name="load")
    pp.create_ext_grid(net, bus=b0, vm_pu=1.0, min_p_mw=-100.0, max_p_mw=100.0, min_q_mvar=-100.0, max_q_mvar=100.0)
    pp.create_line_from_parameters(
        net,
        from_bus=b0,
        to_bus=b1,
        length_km=1.0,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.1,
        c_nf_per_km=10.0,
        max_i_ka=1.0,
    )
    pp.create_gen(
        net,
        bus=b1,
        p_mw=10.0,
        vm_pu=1.0,
        min_p_mw=0.0,
        max_p_mw=20.0,
        min_q_mvar=-50.0,
        max_q_mvar=50.0,
    )
    pp.create_load(net, bus=b1, p_mw=8.0, q_mvar=2.0)
    return net


def _maneuver_payload(vm_pu: float = 1.01) -> dict[str, Any]:
    return {
        "action": {"type": "GEN_V_SETPOINT", "gen_id": 0, "new_vm_pu": vm_pu},
        "diagnosed_cause": "REACTIVE_DEFICIT",
        "rationale": "raise generator voltage",
    }


def _assessment() -> AnalystAssessment:
    return AnalystAssessment(
        diagnosed_cause="REACTIVE_DEFICIT",
        proposed_maneuver=Maneuver.model_validate(_maneuver_payload()),
        rationale="low voltage and Q limits suggest reactive support",
    )


def _config() -> OrchestratorConfig:
    return OrchestratorConfig(
        CONFIGURATION=3,
        MANEUVER_BUDGET=10,
        MAX_RUNTIME_SECONDS=120,
        LLM_ASSIGNMENT=LLMAssignment(single_agent=None, analyst=MODEL_ID, executor=MODEL_ID, orchestrator=MODEL_ID),
    )


def _tool_response(tool_name: str, tool_input: dict[str, Any], tool_use_id: str) -> LLMResponse:
    tool_use = ToolUse(toolUseId=tool_use_id, name=tool_name, input=tool_input)
    assistant_content = ({"toolUse": tool_use.model_dump(mode="json", by_alias=True)},)
    return LLMResponse(
        text="tool call",
        model_id=MODEL_ID,
        tokens_in=101,
        tokens_out=202,
        latency_seconds=0.01,
        raw={"reasoning": "verify the proposed maneuver"},
        tool_use=tool_use,
        assistant_content=assistant_content,
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


def _run_executor(
    monkeypatch: pytest.MonkeyPatch,
    responses: Sequence[LLMResponse],
) -> tuple[Any, ScriptedLLM, Any]:
    fake_llm = ScriptedLLM(responses)
    monkeypatch.setattr(executor, "llm_call", fake_llm)
    grid = create_sandbox(_tiny_net())
    original_vm = float(resolve_net(grid).gen.at[0, "vm_pu"])
    try:
        result = executor.make_executor(MODEL_ID)(
            card="## Scenario Card\nGenerator 0 is at bus 1.",
            grid=grid,
            assessment=_assessment(),
            config=_config(),
        )
        unchanged_vm = float(resolve_net(grid).gen.at[0, "vm_pu"])
    finally:
        discard_sandbox(grid)
    return result, fake_llm, (original_vm, unchanged_vm)


def _message_text(messages: list[ChatMessage]) -> str:
    rendered: list[str] = []
    for message in messages:
        if isinstance(message.content, str):
            rendered.append(message.content)
        else:
            rendered.extend(str(block) for block in message.content)
    return "\n".join(rendered)


def test_executor_tool_loop_answers_applicability_then_returns_report(monkeypatch: pytest.MonkeyPatch) -> None:
    result, fake_llm, _vm = _run_executor(
        monkeypatch,
        [
            _tool_response(
                "get_action_applicability",
                {"action": {"type": "GEN_V_SETPOINT", "gen_id": 0, "new_vm_pu": 1.01}},
                "toolu_app",
            ),
            _tool_response("propose_maneuver", _maneuver_payload(), "toolu_final"),
        ],
    )

    assert isinstance(result.report, ExecutorReport)
    assert result.report.maneuver == Maneuver.model_validate(_maneuver_payload())
    assert result.report.applicability.applicable is True
    assert result.report.pf_result.converged is True
    assert result.llm_responses[0].raw["role"] == "executor"
    assert result.llm_responses[1].raw["role"] == "executor"
    assert fake_llm.requests[1]["messages"][-2].content == list(result.llm_responses[0].assistant_content)
    assert fake_llm.requests[1]["messages"][-1].content[0]["toolResult"]["toolUseId"] == "toolu_app"


def test_executor_report_preview_uses_copy_and_leaves_grid_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    result, _fake_llm, (original_vm, unchanged_vm) = _run_executor(
        monkeypatch, [_tool_response("propose_maneuver", _maneuver_payload(vm_pu=1.01), "toolu_final")]
    )

    assert result.report.pf_result.converged is True
    assert unchanged_vm == original_vm
    assert result.report.maneuver.action.new_vm_pu == 1.01


def test_executor_returns_failed_diagnostic_previews_to_the_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    invalid_preview = _maneuver_payload(vm_pu=1.02)
    invalid_preview["action"]["gen_id"] = 99
    result, _fake_llm, _vm = _run_executor(
        monkeypatch,
        [
            _tool_response("run_ac_pf", {"maneuver": invalid_preview}, "toolu_bad_preview"),
            _tool_response("propose_maneuver", _maneuver_payload(), "toolu_final"),
        ],
    )

    assert len(result.failed_attempts) == 1
    assert result.failed_attempts[0].kind == "PREVIEW_INVALID"
    assert result.failed_attempts[0].maneuver == Maneuver.model_validate(invalid_preview)


def test_executor_prompt_contains_assessment_card_and_action_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    _result, fake_llm, _vm = _run_executor(
        monkeypatch,
        [_tool_response("propose_maneuver", _maneuver_payload(), "toolu_final")],
    )

    text = _message_text(fake_llm.requests[0]["messages"])
    assert "## Scenario Card\nGenerator 0 is at bus 1." in text
    assert "AnalystAssessment" in text
    assert "REACTIVE_DEFICIT" in text
    assert "GEN_V_SETPOINT" in text
    assert "exactly one clipped +/-0.01 step within [0.95, 1.05]" in text
    assert "verify applicability" in text


def _nonexistent_gen_maneuver() -> Maneuver:
    # Schema-valid (vm within bounds) but grid-invalid: gen 99 does not exist in the tiny
    # net. The schema cannot know the grid, so this is exactly the class of maneuver an
    # LLM can emit that only the applicability/apply layer can reject.
    return Maneuver.model_validate(
        {
            "action": {"type": "GEN_V_SETPOINT", "gen_id": 99, "new_vm_pu": 1.02},
            "diagnosed_cause": "REACTIVE_DEFICIT",
            "rationale": "raise generator voltage",
        }
    )


def test_grid_invalid_maneuver_yields_inapplicable_report_not_a_crash() -> None:
    # The orchestrator only catches ValidationError/LLMFailureError; an InvalidActionError
    # leaking from the report would crash resolve() with no ResolutionResponse.
    grid = create_sandbox(_tiny_net())
    try:
        maneuver = _nonexistent_gen_maneuver()
        report = executor.build_executor_report(grid, maneuver)
        original_vm = float(resolve_net(grid).gen.at[0, "vm_pu"])
    finally:
        discard_sandbox(grid)

    assert report.maneuver == maneuver
    assert report.applicability.applicable is False
    assert report.applicability.reason
    assert report.pf_result is not None  # diagnostics of the CURRENT grid, no maneuver applied
    assert original_vm == 1.0  # grid untouched


def test_apply_time_invalid_action_is_absorbed_as_applicability_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Belt and braces: even when the pre-screen passes, an apply-time InvalidActionError
    # must be absorbed into the report, never escape.
    from restorebench.schemas.errors import InvalidActionError
    from restorebench.schemas.topology import ApplicabilityResult

    maneuver = Maneuver.model_validate(_maneuver_payload(vm_pu=1.02))
    monkeypatch.setattr(
        executor,
        "get_action_applicability",
        lambda _grid, action, **_kwargs: ApplicabilityResult(action=action, applicable=True, reason=None),
    )

    def raising_apply(
        _sandbox: Any,
        _maneuver: Maneuver,
        **_kwargs: Any,
    ) -> Any:
        raise InvalidActionError(maneuver.action, "apply-time bound violated")

    monkeypatch.setattr(executor, "apply_maneuver", raising_apply)

    grid = create_sandbox(_tiny_net())
    try:
        report = executor.build_executor_report(grid, maneuver)
    finally:
        discard_sandbox(grid)

    assert report.applicability.applicable is False
    assert "apply-time bound violated" in (report.applicability.reason or "")
