# ABOUTME: Covers the Case-2 single-agent LLM seam without touching Bedrock by default.
# ABOUTME: Verifies prompt construction, tool schema wiring, validation, and reasoning role tagging.
from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import ValidationError

from restorebench.agents import tool_loop
from restorebench.agents.tool_loop import ActionApplicabilityInput
from restorebench.environment.orchestrator import AgentStep, AgentStepResult, resolve
from restorebench.environment.scenarios import load_scenario
from restorebench.llm.models import CHEAPEST_MODEL
from restorebench.llm.providers import ChatMessage, LLMResponse, ToolUse
from restorebench.schemas.actions import Maneuver
from restorebench.schemas.config import LLMAssignment, OrchestratorConfig
from restorebench.schemas.feedback import AcceptedManeuver, FailureFeedback
from restorebench.schemas.power_flow import NRDiagnostics
from restorebench.schemas.topology import ApplicabilityResult
from restorebench.tools.sandbox import create_sandbox, discard_sandbox, resolve_net
from restorebench.corpus.augment import build_augmented_base


MODEL_ID = "test-model"
ACTION_APPLICABILITY_SCHEMA = ActionApplicabilityInput.model_json_schema()
_DIAGNOSTICS_UNSET = object()




def _run_scripted_agent(
    monkeypatch: pytest.MonkeyPatch,
    responses: Sequence[LLMResponse],
    grid: Any,
    *,
    config: OrchestratorConfig | None = None,
    diagnostics: NRDiagnostics | None | object = _DIAGNOSTICS_UNSET,
    failures: Sequence[FailureFeedback] = (),
) -> tuple[AgentStepResult, ScriptedLLM]:
    from restorebench.agents import single_agent

    fake_llm = ScriptedLLM(responses)
    monkeypatch.setattr(single_agent, "llm_call", fake_llm)
    step_diagnostics = _diagnostics() if diagnostics is _DIAGNOSTICS_UNSET else diagnostics
    result = single_agent.make_single_agent(MODEL_ID)(
        card="## Scenario Card\nBus 44 is connected to [45]",
        grid=grid,
        diagnostics=step_diagnostics,
        history=(),
        failures=failures,
        config=config or _config(),
    )
    return result, fake_llm


def _call_agent(
    monkeypatch: pytest.MonkeyPatch,
    response: LLMResponse,
    *,
    diagnostics: NRDiagnostics | None = None,
    history: Sequence[Maneuver] = (),
    failures: Sequence[FailureFeedback] = (),
    config: OrchestratorConfig | None = None,
    grid: Any | None = None,
) -> tuple[AgentStepResult, dict[str, Any]]:
    from restorebench.agents import single_agent

    captured: dict[str, Any] = {}

    def fake_llm_call(
        model_id: str,
        messages: list[ChatMessage],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        thinking: bool = False,
        tools: dict[str, Any] | None = None,
    ) -> LLMResponse:
        captured.update(
            {
                "model_id": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "thinking": thinking,
                "tools": tools,
            }
        )
        return response

    monkeypatch.setattr(single_agent, "llm_call", fake_llm_call)
    agent = single_agent.make_single_agent(MODEL_ID)
    sandbox = grid if grid is not None else create_sandbox({"grid": "present"})
    try:
        result = agent(
            card="## Scenario Card\nBus 44 is connected to [45]",
            grid=sandbox,
            diagnostics=diagnostics,
            history=history,
            failures=failures,
            config=config or _config(),
        )
        return result, captured
    finally:
        if grid is None:
            discard_sandbox(sandbox)


def _maneuver_payload(gen_id: int = 11) -> dict[str, Any]:
    return {
        "action": {"type": "GEN_V_SETPOINT", "gen_id": gen_id, "new_vm_pu": 1.01},
        "diagnosed_cause": "REACTIVE_DEFICIT",
        "rationale": f"raise generator {gen_id} voltage",
    }


def _llm_response(
    tool_input: dict[str, Any] | None = None,
    raw: dict[str, Any] | None = None,
    *,
    tool_name: str = "propose_maneuver",
    tool_use_id: str = "toolu_1",
) -> LLMResponse:
    tool_use = None
    if tool_input is not None:
        tool_use = ToolUse(toolUseId=tool_use_id, name=tool_name, input=tool_input)
    assistant_content = ({"toolUse": tool_use.model_dump(mode="json", by_alias=True)},) if tool_use is not None else ()
    return LLMResponse(
        text="tool call",
        model_id=MODEL_ID,
        tokens_in=101,
        tokens_out=202,
        latency_seconds=0.01,
        raw=raw or {},
        tool_use=tool_use,
        assistant_content=assistant_content,
    )


def _multi_tool_response(tool_uses: list[dict[str, Any]], *, text: str = "tool calls") -> LLMResponse:
    parsed = tuple(ToolUse.model_validate(tool_use) for tool_use in tool_uses)
    assistant_content = tuple({"toolUse": tool_use} for tool_use in tool_uses)
    return LLMResponse(
        text=text,
        model_id=MODEL_ID,
        tokens_in=101,
        tokens_out=202,
        latency_seconds=0.01,
        raw={"assistant_content": list(assistant_content)},
        tool_uses=parsed,
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


def _config(configuration: int = 2) -> OrchestratorConfig:
    return OrchestratorConfig(
        CONFIGURATION=configuration,
        MANEUVER_BUDGET=10,
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


def _tool_result_blocks(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    content = messages[-1].content
    assert isinstance(content, list)
    return [block["toolResult"] for block in content if "toolResult" in block]


def _message_text(messages: list[ChatMessage]) -> str:
    rendered: list[str] = []
    for message in messages:
        if isinstance(message.content, str):
            rendered.append(message.content)
        else:
            rendered.extend(str(block) for block in message.content)
    return "\n".join(rendered)


def test_make_single_agent_returns_agent_step_protocol_callable() -> None:
    from restorebench.agents.single_agent import make_single_agent

    agent: AgentStep = make_single_agent(MODEL_ID)

    assert callable(agent)


def test_agent_returns_validated_maneuver_and_tags_response_role(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _llm_response(_maneuver_payload(), raw={"reasoning": "reactive deficit near bus 45"})

    result, _captured = _call_agent(monkeypatch, response)

    assert isinstance(result, AgentStepResult)
    assert result.maneuver == Maneuver.model_validate(_maneuver_payload())
    assert result.llm_responses == (response,)
    assert response.raw["role"] == "single_agent"


def test_llm_call_uses_generated_maneuver_tool_schema_thinking_and_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _llm_response(_maneuver_payload())

    _result, captured = _call_agent(monkeypatch, response)

    tool_specs = {tool["toolSpec"]["name"]: tool["toolSpec"] for tool in captured["tools"]["tools"]}
    assert set(tool_specs) == {
        "get_grid_topology",
        "get_action_applicability",
        "run_ac_pf",
        "rank_candidate_maneuvers",
        "propose_maneuver",
    }
    assert tool_specs["propose_maneuver"]["inputSchema"]["json"] == Maneuver.model_json_schema()
    assert tool_specs["propose_maneuver"]["inputSchema"]["json"] is not Maneuver.model_json_schema()
    assert tool_specs["get_action_applicability"]["inputSchema"]["json"] == ACTION_APPLICABILITY_SCHEMA
    assert captured["thinking"] is True
    assert captured["temperature"] == 1.0
    assert captured["model_id"] == MODEL_ID


def test_missing_tool_use_raises_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _llm_response(tool_input=None)

    with pytest.raises(ValidationError):
        _call_agent(monkeypatch, response)


def test_invalid_tool_payload_raises_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _llm_response({"action": {"type": "GEN_V_SETPOINT"}, "diagnosed_cause": None, "rationale": ""})

    with pytest.raises(ValidationError):
        _call_agent(monkeypatch, response)


def test_prompt_includes_diagnostics_history_and_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    accepted = Maneuver.model_validate(_maneuver_payload(gen_id=7))
    history = (
        AcceptedManeuver(maneuver=accepted, overstress_before=0.0333, overstress_after=0.0281),
    )
    failures = (
        FailureFeedback(
            iteration=0,
            kind="INVALID_ACTION",
            diagnostics=_diagnostics(),
            detail="generator 7 is Q_LIMITED_UPPER",
            maneuver=accepted,
        ),
    )
    response = _llm_response(_maneuver_payload())

    _result, captured = _call_agent(
        monkeypatch, response, diagnostics=_diagnostics(), history=history, failures=failures
    )

    messages = captured["messages"]
    assert isinstance(messages, list)
    assert all(isinstance(message, ChatMessage) for message in messages)
    system_text = messages[0].content
    user_text = messages[1].content
    assert "GEN_V_SETPOINT" in system_text
    assert "0.95" in system_text and "1.05" in system_text
    assert "TAP_ADJUSTMENT" in system_text
    assert "exactly one +/-1 tap step" in system_text
    assert "REACTIVE_DEFICIT" in system_text
    assert "CORRIDOR_OVERSTRESS" in system_text
    assert "BAD_SETPOINTS" in system_text
    assert "QLIM_INSTABILITY" in system_text
    assert "WEAK_SLACK" in system_text
    assert "Use only the declared atomic Q-V actions" in system_text
    assert "reconstruct" in system_text.lower()
    assert "MANEUVER_BUDGET=10" in system_text
    assert "worst_bus: 44" in user_text
    assert "lowest_vm_pu: 0.71" in user_text
    assert "gens_at_q_limit: [1, 2]" in user_text
    assert "max_mismatch_mw: 12.5" in user_text
    assert "GEN_V_SETPOINT" in user_text
    assert "gen_id" in user_text and "7" in user_text
    assert "Failed attempts on current grid state (do not repeat their actions)" in user_text
    assert "INVALID_ACTION" in user_text
    assert "generator 7 is Q_LIMITED_UPPER" in user_text


def test_shared_prompt_fragments_preserve_single_agent_system_prompt() -> None:
    from restorebench.agents import single_agent
    from restorebench.agents.prompt_fragments import (
        ACTION_VOCABULARY_AND_BOUNDS,
        CAUSE_ACTION_PRIOR,
        CAUSE_TAXONOMY,
        COMPOSITION_AND_PROGRESS,
    )

    expected = """You are the Case-2 single grid-resolution agent. Propose exactly one Maneuver.

First reconstruct the relevant grid structure from the Scenario Card before choosing the maneuver.
The orchestrator will apply the maneuver, run the locked AC power flow, and decide convergence.
Do not propose topology switching, load changes, adding/removing components, or more than one action.

Allowed actions and bounds:
- GEN_V_SETPOINT: gen_id, new_vm_pu exactly one clipped +/-0.01 step within [0.95, 1.05]. If valid diagnostics mark the generator Q_LIMITED_UPPER, only a decrease is available; if Q_LIMITED_LOWER, only an increase is available.
- SHUNT_STEP: shunt_id, new_step toggling one available single-step shunt between 0 and 1. q_mvar < 0 is a capacitor/injects reactive power; q_mvar > 0 is a reactor/absorbs.
- TAP_ADJUSTMENT: trafo_id, new_tap_pos exactly one +/-1 tap step within the declared bounds.

Cause taxonomy:
- REACTIVE_DEFICIT: insufficient reactive support near the P-V nose; low load-bus voltages; generators saturated at max_q_mvar.
- CORRIDOR_OVERSTRESS: a heavily loaded import corridor cannot be solved at the demanded transfer.
- BAD_SETPOINTS: generator voltage setpoints inconsistent, too low, or too high for the operating point.
- QLIM_INSTABILITY: PV-to-PQ switching at Q limits prevents the enforced-Q-limit PF from settling.
- WEAK_SLACK: the slack is asked to source or sink an infeasible amount; redispatch toward generators.

Cause to action prior:
- REACTIVE_DEFICIT: available remedies are GEN_V_SETPOINT, SHUNT_STEP, and TAP_ADJUSTMENT.
- CORRIDOR_OVERSTRESS: only the declared Q-V controls are available in v0.
- BAD_SETPOINTS: strongest remedies are GEN_V_SETPOINT and TAP_ADJUSTMENT.
- QLIM_INSTABILITY: strongest remedy is SHUNT_STEP; on a Q_LIMITED_UPPER generator only a setpoint decrease is allowed.
- WEAK_SLACK: only the declared Q-V controls are available in v0.

Composing maneuvers:
- Accepted maneuvers persist. Each one stays applied to the grid, and the Scenario Card and diagnostics you receive next already reflect every maneuver accepted so far. You are extending one cumulative plan across iterations, not starting over each time.
- 'overstress' in the NR diagnostics measures how far the case sits beyond the solvable boundary: it is the current load divided by the largest load this grid can still solve, minus one. Lower is closer to convergence and zero means solvable.
- A maneuver that lowers overstress is progress even when the grid still diverges. Keep it and build the next maneuver on top of it. Many cases admit no single action that converges and are solvable only by several stacked maneuvers.
- When no candidate converges, choose the one with the lowest overstress. A candidate that raises overstress is moving the wrong way.

Diagnostic tools available before the final maneuver: rank_candidate_maneuvers, get_grid_topology, get_action_applicability, and run_ac_pf.
Start each iteration with rank_candidate_maneuvers: it evaluates a spread of legal maneuvers with real power flows and tells you which ones remove overstress. Propose its top candidate unless you have a specific reason not to; use run_ac_pf only to check something the ranking did not cover.
run_ac_pf is your verification tool: pass a candidate maneuver and read 'converged' to see if it fixes the divergence. Test promising candidates with it before proposing; calling it with no maneuver only re-previews the known-diverging base case.
Use at most 8 diagnostic tool calls. The final answer must be a propose_maneuver tool call.
Use only the declared atomic Q-V actions.
MANEUVER_BUDGET=10; this call proposes one maneuver for the current iteration."""

    shared = "\n\n".join(
        [ACTION_VOCABULARY_AND_BOUNDS, CAUSE_TAXONOMY, CAUSE_ACTION_PRIOR, COMPOSITION_AND_PROGRESS]
    )

    assert shared in single_agent._system_prompt(_config())
    assert single_agent._system_prompt(_config()) == expected


def test_tool_loop_answers_applicability_result_and_preserves_assistant_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    grid = create_sandbox({"diagnostic": "grid"})
    action = {"type": "GEN_V_SETPOINT", "gen_id": 11, "new_vm_pu": 1.01}
    tool_use = {"toolUseId": "toolu_app", "name": "get_action_applicability", "input": {"action": action}}
    first = _multi_tool_response([tool_use])
    second = _llm_response(_maneuver_payload())

    def fake_applicability(seen_grid: Any, seen_action: Any, **_kwargs: Any) -> ApplicabilityResult:
        assert seen_grid == grid
        return ApplicabilityResult(action=seen_action, applicable=True, reason=None)

    monkeypatch.setattr(tool_loop, "get_action_applicability", fake_applicability)
    try:
        result, fake_llm = _run_scripted_agent(monkeypatch, [first, second], grid)
    finally:
        discard_sandbox(grid)

    assert result.maneuver == Maneuver.model_validate(_maneuver_payload())
    assert result.llm_responses == (first, second)
    assert fake_llm.requests[1]["messages"][-2].content == list(first.assistant_content)
    tool_results = _tool_result_blocks(fake_llm.requests[1]["messages"])
    assert tool_results == [
        {
            "toolUseId": "toolu_app",
            "content": [{"json": {"action": action, "applicable": True, "reason": None}}],
            "status": "success",
        }
    ]


def test_run_ac_pf_preview_with_maneuver_uses_copy_and_leaves_grid_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    net = build_augmented_base()
    gen_id = int(net.gen.index[net.gen["in_service"].astype(bool)][0])
    original_vm_pu = float(net.gen.at[gen_id, "vm_pu"])
    new_vm_pu = round(original_vm_pu + 0.01, 10) if original_vm_pu < 1.05 else round(original_vm_pu - 0.01, 10)
    grid = create_sandbox(net)
    preview_maneuver = _maneuver_payload(gen_id=gen_id)
    preview_maneuver["action"]["new_vm_pu"] = new_vm_pu
    first = _multi_tool_response(
        [{"toolUseId": "toolu_pf", "name": "run_ac_pf", "input": {"maneuver": preview_maneuver}}]
    )
    second = _llm_response(_maneuver_payload())

    try:
        result, fake_llm = _run_scripted_agent(monkeypatch, [first, second], grid)
        unchanged = float(resolve_net(grid).gen.at[gen_id, "vm_pu"])
    finally:
        discard_sandbox(grid)

    tool_results = _tool_result_blocks(fake_llm.requests[1]["messages"])
    assert result.maneuver == Maneuver.model_validate(_maneuver_payload())
    assert tool_results[0]["status"] == "success"
    assert "converged" in tool_results[0]["content"][0]["json"]
    assert unchanged == pytest.approx(original_vm_pu)


def test_invalid_preview_action_returns_error_tool_result_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    net = build_augmented_base()
    gen_id = int(net.gen.index[(net.gen["min_p_mw"] < net.gen["max_p_mw"]) & net.gen["in_service"].astype(bool)][0])
    current_vm_pu = float(net.gen.at[gen_id, "vm_pu"])
    invalid_target = round(current_vm_pu - 0.02 if current_vm_pu >= 0.97 else current_vm_pu + 0.02, 10)
    grid = create_sandbox(net)
    invalid_maneuver = {
        "action": {
            "type": "GEN_V_SETPOINT",
            "gen_id": gen_id,
            "new_vm_pu": invalid_target,
        },
        "diagnosed_cause": "REACTIVE_DEFICIT",
        "rationale": "probe a non-atomic voltage target",
    }
    first = _multi_tool_response(
        [{"toolUseId": "toolu_bad_pf", "name": "run_ac_pf", "input": {"maneuver": invalid_maneuver}}]
    )
    second = _llm_response(_maneuver_payload())

    try:
        result, fake_llm = _run_scripted_agent(monkeypatch, [first, second], grid)
    finally:
        discard_sandbox(grid)

    tool_results = _tool_result_blocks(fake_llm.requests[1]["messages"])
    assert result.maneuver == Maneuver.model_validate(_maneuver_payload())
    assert len(result.failed_attempts) == 1
    assert result.failed_attempts[0].kind == "PREVIEW_INVALID"
    assert result.failed_attempts[0].maneuver == Maneuver.model_validate(invalid_maneuver)
    assert tool_results[0]["status"] == "error"
    assert "atomic" in tool_results[0]["content"][0]["text"]


def test_failed_preview_cannot_be_repeated_in_the_same_tool_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    net = build_augmented_base()
    gen_id = int(net.gen.index[(net.gen["min_p_mw"] < net.gen["max_p_mw"]) & net.gen["in_service"].astype(bool)][0])
    current_vm_pu = float(net.gen.at[gen_id, "vm_pu"])
    invalid_target = round(current_vm_pu - 0.02 if current_vm_pu >= 0.97 else current_vm_pu + 0.02, 10)
    grid = create_sandbox(net)
    invalid_maneuver = {
        "action": {
            "type": "GEN_V_SETPOINT",
            "gen_id": gen_id,
            "new_vm_pu": invalid_target,
        },
        "diagnosed_cause": "REACTIVE_DEFICIT",
        "rationale": "probe the same non-atomic voltage target",
    }
    first = _multi_tool_response(
        [{"toolUseId": "toolu_bad_pf_1", "name": "run_ac_pf", "input": {"maneuver": invalid_maneuver}}]
    )
    second = _multi_tool_response(
        [{"toolUseId": "toolu_bad_pf_2", "name": "run_ac_pf", "input": {"maneuver": invalid_maneuver}}]
    )
    third = _llm_response(_maneuver_payload())

    try:
        result, fake_llm = _run_scripted_agent(monkeypatch, [first, second, third], grid)
    finally:
        discard_sandbox(grid)

    repeated_result = _tool_result_blocks(fake_llm.requests[2]["messages"])[0]
    assert repeated_result["status"] == "error"
    assert "already failed on the current grid state" in repeated_result["content"][0]["text"]
    assert len(result.failed_attempts) == 1


def test_failed_preview_from_prior_iteration_is_blocked_on_unchanged_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    net = build_augmented_base()
    gen_id = int(net.gen.index[net.gen["in_service"].astype(bool)][0])
    grid = create_sandbox(net)
    failed_maneuver = Maneuver.model_validate(_maneuver_payload(gen_id=gen_id))
    failures = (
        FailureFeedback(
            iteration=0,
            kind="PREVIEW_DIVERGED",
            diagnostics=_diagnostics(),
            detail="preview did not converge",
            maneuver=failed_maneuver,
        ),
    )
    first = _multi_tool_response(
        [
            {
                "toolUseId": "toolu_repeated_pf",
                "name": "run_ac_pf",
                "input": {"maneuver": failed_maneuver.model_dump(mode="json")},
            }
        ]
    )
    second = _llm_response(_maneuver_payload())

    try:
        result, fake_llm = _run_scripted_agent(
            monkeypatch,
            [first, second],
            grid,
            failures=failures,
        )
    finally:
        discard_sandbox(grid)

    repeated_result = _tool_result_blocks(fake_llm.requests[1]["messages"])[0]
    assert repeated_result["status"] == "error"
    assert "already failed on the current grid state" in repeated_result["content"][0]["text"]
    assert result.failed_attempts == ()


def test_budget_exhaustion_warns_once_then_missing_proposal_raises_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from restorebench.agents import single_agent

    from restorebench.agents.tool_loop import MAX_DIAGNOSTIC_TOOL_CALLS

    grid = create_sandbox({"diagnostic": "grid"})
    # Exhaust the budget: one diagnostic-only turn per allowed call, then one more that must propose.
    responses = [
        _multi_tool_response([{"toolUseId": f"toolu_{i}", "name": "get_grid_topology", "input": {}}])
        for i in range(MAX_DIAGNOSTIC_TOOL_CALLS + 1)
    ]
    fake_llm = ScriptedLLM(responses)
    monkeypatch.setattr(single_agent, "llm_call", fake_llm)
    monkeypatch.setattr(
        tool_loop,
        "get_grid_topology",
        lambda _grid, **_kwargs: {"topology": "summary"},
    )

    try:
        with pytest.raises(ValidationError):
            single_agent.make_single_agent(MODEL_ID)(
                card="card",
                grid=grid,
                diagnostics=_diagnostics(),
                history=(),
                failures=(),
                config=_config(),
            )
    finally:
        discard_sandbox(grid)

    assert "tool budget is exhausted" in _message_text(fake_llm.requests[MAX_DIAGNOSTIC_TOOL_CALLS]["messages"]).lower()


def test_multiple_diagnostic_tool_uses_are_all_answered(monkeypatch: pytest.MonkeyPatch) -> None:

    grid = create_sandbox({"diagnostic": "grid"})
    action = {"type": "SHUNT_STEP", "shunt_id": 0, "new_step": 0}
    first = _multi_tool_response(
        [
            {"toolUseId": "toolu_topology", "name": "get_grid_topology", "input": {}},
            {"toolUseId": "toolu_applicability", "name": "get_action_applicability", "input": {"action": action}},
        ]
    )
    second = _llm_response(_maneuver_payload())

    monkeypatch.setattr(
        tool_loop,
        "get_grid_topology",
        lambda _grid, **_kwargs: {"topology": "summary"},
    )
    monkeypatch.setattr(
        tool_loop,
        "get_action_applicability",
        lambda _grid, parsed_action, **_kwargs: ApplicabilityResult(action=parsed_action, applicable=True, reason=None),
    )
    try:
        _result, fake_llm = _run_scripted_agent(monkeypatch, [first, second], grid)
    finally:
        discard_sandbox(grid)

    tool_results = _tool_result_blocks(fake_llm.requests[1]["messages"])
    assert [result["toolUseId"] for result in tool_results] == ["toolu_topology", "toolu_applicability"]
    assert all(result["status"] == "success" for result in tool_results)


def test_grid_none_is_loud_wiring_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from restorebench.agents import single_agent

    fake_llm = ScriptedLLM([_llm_response(_maneuver_payload())])
    monkeypatch.setattr(single_agent, "llm_call", fake_llm)

    with pytest.raises(RuntimeError, match="diagnostic grid"):
        single_agent.make_single_agent(MODEL_ID)(
            card="card",
            grid=None,
            diagnostics=_diagnostics(),
            history=(),
            failures=(),
            config=_config(),
        )

    assert fake_llm.requests == []


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RESTOREBENCH_LLM_INTEGRATION") != "1",
    reason="real Bedrock single-agent integration is opt-in and not part of pre-push",
)
def test_live_single_agent_with_orchestrator_records_reasoning() -> None:
    from restorebench.agents.single_agent import make_single_agent

    scenario = load_scenario("S0121")
    config = OrchestratorConfig(
        CONFIGURATION=2,
        MANEUVER_BUDGET=1,
        MAX_RUNTIME_SECONDS=300,
        LLM_ASSIGNMENT=LLMAssignment(
            single_agent=CHEAPEST_MODEL,
            analyst=None,
            executor=None,
            orchestrator=None,
        ),
    )

    response = resolve(scenario, config, make_single_agent(CHEAPEST_MODEL))

    assert response.maneuvers
    assert all(isinstance(maneuver, Maneuver) for maneuver in response.maneuvers)
    assert response.trace.reasoning
    assert {entry.role for entry in response.trace.reasoning} == {"single_agent"}
