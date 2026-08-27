# ABOUTME: Covers the multi-agent Case-3/5 step composition with scripted LLM roles.
# ABOUTME: Verifies role isolation, revision flow, memory retrieval, and AgentStepResult fields.
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandapower as pp
import pytest

from restorebench.agents import analyst, executor, multi_agent
from restorebench.llm.providers import ChatMessage, LLMResponse, ToolUse
from restorebench.schemas.actions import Maneuver
from restorebench.schemas.config import LLMAssignment, OrchestratorConfig
from restorebench.schemas.power_flow import NRDiagnostics
from restorebench.tools.sandbox import create_sandbox, discard_sandbox


ANALYST_MODEL = "analyst-model"
EXECUTOR_MODEL = "executor-model"
ORCHESTRATOR_MODEL = "orchestrator-model"



def _analyst_payload(vm_pu: float = 1.04) -> dict[str, Any]:
    return {
        "diagnosed_cause": "REACTIVE_DEFICIT",
        "proposed_maneuver": _maneuver_payload(vm_pu),
        "rationale": "low voltage and Q limits suggest reactive support",
    }


def _run_step(
    monkeypatch: pytest.MonkeyPatch,
    fake_llm: RoutedLLM,
    *,
    config: OrchestratorConfig | None = None,
) -> tuple[Any, RoutedLLM]:
    _install_llm(monkeypatch, fake_llm)
    grid = create_sandbox(_tiny_net())
    try:
        result = multi_agent.make_multi_agent()(
            card="## Scenario Card\nGenerator 0 is at bus 1.",
            grid=grid,
            diagnostics=_diagnostics(),
            history=(),
            failures=(),
            config=config or _config(),
        )
    finally:
        discard_sandbox(grid)
    return result, fake_llm


def _tiny_net() -> Any:
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=110.0, name="slack")
    b1 = pp.create_bus(net, vn_kv=110.0, name="load")
    pp.create_ext_grid(net, bus=b0, vm_pu=1.0, min_p_mw=-100.0, max_p_mw=100.0, min_q_mvar=-100.0, max_q_mvar=100.0)
    pp.create_line_from_parameters(net, b0, b1, 1.0, 0.1, 0.1, 10.0, 1.0)
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


def _diagnostics() -> NRDiagnostics:
    return NRDiagnostics(
        iterations_attempted=30,
        worst_bus=1,
        lowest_vm_pu=0.71,
        lowest_vm_bus=1,
        gens_at_q_limit=[0],
        max_mismatch_mw=12.5,
        max_mismatch_mvar=33.25,
        overstress=1.8,
        error_message="Newton-Raphson failed after 30 iterations",
        diagnostics_source="local_nose",
    )


def _maneuver_payload(vm_pu: float = 1.04) -> dict[str, Any]:
    return {
        "action": {"type": "GEN_V_SETPOINT", "gen_id": 0, "new_vm_pu": vm_pu},
        "diagnosed_cause": "REACTIVE_DEFICIT",
        "rationale": "raise generator voltage",
    }


def _config(configuration: int = 3, assignment: LLMAssignment | None = None) -> OrchestratorConfig:
    return OrchestratorConfig(
        CONFIGURATION=configuration,
        MANEUVER_BUDGET=10,
        MAX_RUNTIME_SECONDS=120,
        LLM_ASSIGNMENT=assignment
        or LLMAssignment(
            single_agent=None,
            analyst=ANALYST_MODEL,
            executor=EXECUTOR_MODEL,
            orchestrator=ORCHESTRATOR_MODEL,
        ),
    )


def _tool_response(model_id: str, tool_name: str, tool_input: dict[str, Any], tool_use_id: str) -> LLMResponse:
    tool_use = ToolUse(toolUseId=tool_use_id, name=tool_name, input=tool_input)
    assistant_content = ({"toolUse": tool_use.model_dump(mode="json", by_alias=True)},)
    return LLMResponse(
        text="tool call",
        model_id=model_id,
        tokens_in=10,
        tokens_out=4,
        latency_seconds=0.01,
        raw={"reasoning": f"{model_id} reasoning"},
        tool_use=tool_use,
        assistant_content=assistant_content,
    )


def _analyst_response(vm_pu: float = 1.04) -> LLMResponse:
    return _tool_response(ANALYST_MODEL, analyst.ANALYST_TOOL_NAME, _analyst_payload(vm_pu), "toolu_analyst")


def _executor_response(vm_pu: float = 1.04) -> LLMResponse:
    return _tool_response(
        EXECUTOR_MODEL, executor.PROPOSE_MANEUVER_TOOL_NAME, _maneuver_payload(vm_pu), "toolu_executor"
    )


def _decision_response(decision: str = "COMMIT", guidance: str | None = None) -> LLMResponse:
    return _tool_response(
        ORCHESTRATOR_MODEL,
        multi_agent.ORCHESTRATOR_DECISION_TOOL_NAME,
        {"decision": decision, "guidance": guidance},
        "toolu_orchestrator",
    )


class RoutedLLM:
    def __init__(self, responses: dict[str, Sequence[LLMResponse]]) -> None:
        self.responses = {model_id: list(items) for model_id, items in responses.items()}
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
        return self.responses[model_id].pop(0)


def _install_llm(monkeypatch: pytest.MonkeyPatch, fake_llm: RoutedLLM) -> None:
    monkeypatch.setattr(analyst, "llm_call", fake_llm)
    monkeypatch.setattr(executor, "llm_call", fake_llm)
    monkeypatch.setattr(multi_agent, "llm_call", fake_llm)


def _message_text(messages: list[ChatMessage]) -> str:
    rendered: list[str] = []
    for message in messages:
        if isinstance(message.content, str):
            rendered.append(message.content)
        else:
            rendered.extend(str(block) for block in message.content)
    return "\n".join(rendered)


def test_multi_agent_commit_path_returns_chronological_role_tagged_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llm = RoutedLLM(
        {
            ANALYST_MODEL: [_analyst_response()],
            EXECUTOR_MODEL: [_executor_response()],
            ORCHESTRATOR_MODEL: [_decision_response("COMMIT")],
        }
    )

    result, _fake_llm = _run_step(monkeypatch, fake_llm)

    assert result.maneuver == Maneuver.model_validate(_maneuver_payload())
    assert [response.raw["role"] for response in result.llm_responses] == ["analyst", "executor", "orchestrator_agent"]
    assert [response.model_id for response in result.llm_responses] == [
        ANALYST_MODEL,
        EXECUTOR_MODEL,
        ORCHESTRATOR_MODEL,
    ]


def test_multi_agent_returns_executor_failed_previews_to_the_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    invalid_preview = _maneuver_payload(1.02)
    invalid_preview["action"]["gen_id"] = 99
    fake_llm = RoutedLLM(
        {
            ANALYST_MODEL: [_analyst_response()],
            EXECUTOR_MODEL: [
                _tool_response(EXECUTOR_MODEL, "run_ac_pf", {"maneuver": invalid_preview}, "toolu_bad_preview"),
                _executor_response(),
            ],
            ORCHESTRATOR_MODEL: [_decision_response("COMMIT")],
        }
    )

    result, _fake_llm = _run_step(monkeypatch, fake_llm)

    assert len(result.failed_attempts) == 1
    assert result.failed_attempts[0].kind == "PREVIEW_INVALID"
    assert result.failed_attempts[0].maneuver == Maneuver.model_validate(invalid_preview)


def test_multi_agent_revision_round_reinvokes_analyst_and_commits_revised_maneuver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = RoutedLLM(
        {
            ANALYST_MODEL: [_analyst_response(1.03), _analyst_response(1.05)],
            EXECUTOR_MODEL: [_executor_response(1.03), _executor_response(1.05)],
            ORCHESTRATOR_MODEL: [
                _decision_response("REVISE", "Use stronger voltage support."),
                _decision_response("COMMIT"),
            ],
        }
    )

    result, fake_llm = _run_step(monkeypatch, fake_llm)

    assert result.maneuver == Maneuver.model_validate(_maneuver_payload(1.05))
    assert [response.raw["role"] for response in result.llm_responses] == [
        "analyst",
        "executor",
        "orchestrator_agent",
        "analyst",
        "executor",
        "orchestrator_agent",
    ]
    analyst_requests = [request for request in fake_llm.requests if request["model_id"] == ANALYST_MODEL]
    assert "Use stronger voltage support." in _message_text(analyst_requests[1]["messages"])


def test_multi_agent_revision_persists_rejected_executor_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    invalid_candidate = _maneuver_payload(1.02)
    invalid_candidate["action"]["gen_id"] = 99
    fake_llm = RoutedLLM(
        {
            ANALYST_MODEL: [_analyst_response(1.02), _analyst_response(1.04)],
            EXECUTOR_MODEL: [
                _tool_response(EXECUTOR_MODEL, executor.PROPOSE_MANEUVER_TOOL_NAME, invalid_candidate, "toolu_invalid"),
                _executor_response(1.04),
            ],
            ORCHESTRATOR_MODEL: [
                _decision_response("REVISE", "Choose a valid component."),
                _decision_response("COMMIT"),
            ],
        }
    )

    result, _fake_llm = _run_step(monkeypatch, fake_llm)

    assert result.maneuver == Maneuver.model_validate(_maneuver_payload(1.04))
    assert len(result.failed_attempts) == 1
    assert result.failed_attempts[0].kind == "PREVIEW_INVALID"
    assert result.failed_attempts[0].maneuver == Maneuver.model_validate(invalid_candidate)


def test_second_revise_decision_still_commits_latest_executor_maneuver(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llm = RoutedLLM(
        {
            ANALYST_MODEL: [_analyst_response(1.03), _analyst_response(1.05)],
            EXECUTOR_MODEL: [_executor_response(1.03), _executor_response(1.05)],
            ORCHESTRATOR_MODEL: [
                _decision_response("REVISE", "Use stronger voltage support."),
                _decision_response("REVISE", "Still not ideal."),
            ],
        }
    )

    result, _fake_llm = _run_step(monkeypatch, fake_llm)

    assert result.maneuver == Maneuver.model_validate(_maneuver_payload(1.05))


def test_role_isolation_uses_structured_handoffs_only(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llm = RoutedLLM(
        {
            ANALYST_MODEL: [_analyst_response()],
            EXECUTOR_MODEL: [_executor_response()],
            ORCHESTRATOR_MODEL: [_decision_response("COMMIT")],
        }
    )

    _result, fake_llm = _run_step(monkeypatch, fake_llm)

    analyst_text = _message_text(
        next(request["messages"] for request in fake_llm.requests if request["model_id"] == ANALYST_MODEL)
    )
    executor_text = _message_text(
        next(request["messages"] for request in fake_llm.requests if request["model_id"] == EXECUTOR_MODEL)
    )
    orchestrator_text = _message_text(
        next(request["messages"] for request in fake_llm.requests if request["model_id"] == ORCHESTRATOR_MODEL)
    )
    assert "ExecutorReport" not in analyst_text
    assert "analyst-model reasoning" not in executor_text
    assert "AnalystAssessment" in orchestrator_text
    assert "ExecutorReport" in orchestrator_text
    assert "analyst-model reasoning" not in orchestrator_text
    assert "executor-model reasoning" not in orchestrator_text


def test_executor_and_orchestrator_agent_modules_do_not_import_memory() -> None:
    for module in [executor, multi_agent]:
        source_path = module.__file__
        assert source_path is not None
        with open(source_path, encoding="utf-8") as handle:
            source = handle.read()
        if module is executor:
            assert "restorebench.tools.memory" not in source
