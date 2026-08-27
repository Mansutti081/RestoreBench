# ABOUTME: Verifies the deterministic resolution runner around scripted agent seams.
# ABOUTME: Covers loop status mapping, trace counters, Case 1 applier, and corpus integrity.
from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pandapower as pp
from pathlib import Path

import pytest

from restorebench.agents import analyst as analyst_agent
from restorebench.agents import baseline_chatbot
from restorebench.agents import executor as executor_agent
from restorebench.agents import multi_agent
from restorebench.environment import orchestrator as orch
from restorebench.environment.scenarios import load_scenario
from restorebench.llm.providers import ChatMessage, LLMResponse, ToolUse
from restorebench.physics.actions import ACTION_POLICY_VERSION
from restorebench.physics.policies import RANKING_POLICY_VERSION, SOLVER_PROBE_POLICY_VERSION
from restorebench.schemas.actions import Maneuver, ManeuverSequence
from restorebench.schemas.config import LLMAssignment, OrchestratorConfig
from restorebench.schemas.dataset import Scenario
from restorebench.schemas.errors import CorpusIntegrityError, InvalidActionError, LLMFailureError
from restorebench.schemas.feedback import FailedManeuverAttempt
from restorebench.schemas.physics import SolvedFeasibility, VoltageEnvelope
from restorebench.schemas.power_flow import NRDiagnostics, PowerFlowResult, QualityResult
from restorebench.schemas.response import RESULT_SCHEMA_VERSION
from restorebench.tools import sandbox as sandbox_tools


DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset/ieee118"


def _assignment() -> LLMAssignment:
    return LLMAssignment(single_agent="test-model", analyst=None, executor=None, orchestrator=None)


def _multi_assignment() -> LLMAssignment:
    return LLMAssignment(
        single_agent=None,
        analyst="analyst-model",
        executor="executor-model",
        orchestrator="orchestrator-model",
    )


def _config(configuration: int = 2, budget: int = 2, max_runtime_seconds: int = 10) -> OrchestratorConfig:
    return OrchestratorConfig(
        CONFIGURATION=configuration,
        MANEUVER_BUDGET=budget,
        MAX_RUNTIME_SECONDS=max_runtime_seconds,
        LLM_ASSIGNMENT=_assignment(),
    )


def _multi_config(configuration: int = 3, budget: int = 2) -> OrchestratorConfig:
    return OrchestratorConfig(
        CONFIGURATION=configuration,
        MANEUVER_BUDGET=budget,
        MAX_RUNTIME_SECONDS=120,
        LLM_ASSIGNMENT=_multi_assignment(),
    )


def _maneuver(gen_id: int = 11, vm_pu: float = 1.01) -> Maneuver:
    return Maneuver(
        action={"type": "GEN_V_SETPOINT", "gen_id": gen_id, "new_vm_pu": vm_pu},
        diagnosed_cause="REACTIVE_DEFICIT",
        rationale="Raise voltage support.",
    )


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


def _llm_response(tokens_in: int = 10, tokens_out: int = 4, raw: dict[str, Any] | None = None) -> LLMResponse:
    return LLMResponse(
        text="tool call",
        model_id="test-model",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_seconds=0.01,
        raw=raw or {},
    )


def _multi_maneuver_payload(vm_pu: float = 1.01) -> dict[str, Any]:
    return {
        "action": {"type": "GEN_V_SETPOINT", "gen_id": 0, "new_vm_pu": vm_pu},
        "diagnosed_cause": "REACTIVE_DEFICIT",
        "rationale": "Raise voltage support.",
    }


def _multi_tool_response(model_id: str, tool_name: str, tool_input: dict[str, Any], tool_use_id: str) -> LLMResponse:
    tool_use = ToolUse(toolUseId=tool_use_id, name=tool_name, input=tool_input)
    return LLMResponse(
        text="tool call",
        model_id=model_id,
        tokens_in=10,
        tokens_out=4,
        latency_seconds=0.01,
        raw={"reasoning": f"{model_id} reasoning"},
        tool_use=tool_use,
        assistant_content=({"toolUse": tool_use.model_dump(mode="json", by_alias=True)},),
    )


def _multi_analyst_payload(vm_pu: float = 1.01) -> dict[str, Any]:
    return {
        "diagnosed_cause": "REACTIVE_DEFICIT",
        "proposed_maneuver": _multi_maneuver_payload(vm_pu),
        "rationale": "reactive support",
    }


def _multi_analyst_response(vm_pu: float = 1.01) -> LLMResponse:
    return _multi_tool_response(
        "analyst-model", analyst_agent.ANALYST_TOOL_NAME, _multi_analyst_payload(vm_pu), "toolu_analyst"
    )


def _multi_executor_response(vm_pu: float = 1.01) -> LLMResponse:
    return _multi_tool_response(
        "executor-model",
        executor_agent.PROPOSE_MANEUVER_TOOL_NAME,
        _multi_maneuver_payload(vm_pu),
        "toolu_executor",
    )


def _multi_decision_response(decision: str = "COMMIT") -> LLMResponse:
    return _multi_tool_response(
        "orchestrator-model",
        multi_agent.ORCHESTRATOR_DECISION_TOOL_NAME,
        {"decision": decision, "guidance": None},
        "toolu_orchestrator",
    )


class RoutedLLM:
    def __init__(self, responses: dict[str, list[LLMResponse]]) -> None:
        self.responses = {model_id: list(items) for model_id, items in responses.items()}

    def __call__(
        self,
        model_id: str,
        _messages: list[ChatMessage],
        **_kwargs: Any,
    ) -> LLMResponse:
        return self.responses[model_id].pop(0)


def _install_multi_agent_llm(monkeypatch: pytest.MonkeyPatch, fake_llm: RoutedLLM) -> None:
    monkeypatch.setattr(analyst_agent, "llm_call", fake_llm)
    monkeypatch.setattr(executor_agent, "llm_call", fake_llm)
    monkeypatch.setattr(multi_agent, "llm_call", fake_llm)


def _install_tiny_runner(monkeypatch: pytest.MonkeyPatch, pf_results: list[PowerFlowResult]) -> None:
    remaining = list(pf_results)
    monkeypatch.setattr(orch, "load_full_net", lambda _scenario: _tiny_net())
    monkeypatch.setattr(orch, "load_card", lambda _scenario: "tiny scenario card")
    monkeypatch.setattr(orch, "run_ac_pf", lambda _net: remaining.pop(0))


def _sequence_llm_response(maneuvers: list[Maneuver]) -> LLMResponse:
    payload = {
        "maneuvers": [maneuver.model_dump(mode="json") for maneuver in maneuvers],
        "reconstruction_summary": "candidate sequence",
    }
    tool_use = ToolUse(toolUseId="toolu_sequence", name=baseline_chatbot.PROPOSE_SEQUENCE_TOOL_NAME, input=payload)
    return LLMResponse(
        text="tool call",
        model_id="test-model",
        tokens_in=10,
        tokens_out=4,
        latency_seconds=0.01,
        raw={"reasoning": "build graph and try sequence"},
        tool_use=tool_use,
        assistant_content=({"toolUse": tool_use.model_dump(mode="json", by_alias=True)},),
    )


def _diagnostics(bus: int) -> NRDiagnostics:
    return NRDiagnostics(
        iterations_attempted=30,
        worst_bus=bus,
        lowest_vm_pu=0.7,
        lowest_vm_bus=bus,
        gens_at_q_limit=[1, 2],
        error_message=f"diverged at {bus}",
        diagnostics_source="local_nose",
    )


def _pf_diverged(bus: int) -> PowerFlowResult:
    return PowerFlowResult(
        converged=False,
        iterations=30,
        tolerance_used=1e-6,
        runtime_ms=1.0,
        error_message="did not converge",
        diagnostics=_diagnostics(bus),
    )


def _quality() -> QualityResult:
    return QualityResult(
        clean=True,
        n_buses_out_of_band=0,
        worst_vm_pu=0.99,
        worst_vm_bus=10,
        symptoms=[],
    )


def _terminal_feasibility(
    *,
    non_voltage_ok: bool = True,
    hard_envelope_ok: bool = False,
) -> SolvedFeasibility:
    return SolvedFeasibility(
        feasible=non_voltage_ok and hard_envelope_ok,
        generator_p_within_limits=True,
        generator_q_within_limits=True,
        external_grid_within_limits=non_voltage_ok,
        connected=True,
        loads_energized=True,
        voltage=VoltageEnvelope(
            min_vm_pu=0.70 if not hard_envelope_ok else 0.95,
            max_vm_pu=1.05,
            low_bus_ids=(() if hard_envelope_ok else (1,)),
            high_bus_ids=(),
            hard_envelope_ok=hard_envelope_ok,
            runtime_quality_ok=hard_envelope_ok,
        ),
        generator_q_status=(),
        slack_results=(),
        q_limited_gen_ids=(),
        failure_reasons=(),
        policy_version="test-feasibility",
    )


def _pf_converged(*, non_voltage_ok: bool = True) -> PowerFlowResult:
    return PowerFlowResult(
        converged=True,
        iterations=6,
        tolerance_used=1e-8,
        runtime_ms=1.0,
        quality=_quality(),
        warnings=[],
        feasibility=_terminal_feasibility(non_voltage_ok=non_voltage_ok),
    )


def _install_fake_environment(
    monkeypatch: pytest.MonkeyPatch,
    pf_results: list[PowerFlowResult],
    *,
    apply_hook: Callable[[Any, Maneuver], None] | None = None,
    initial_working: Any = "net0",
    resolved_values: list[Any] | None = None,
    render_card: Callable[[Any], str] | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "run_inputs": [],
        "create_inputs": [],
        "diagnostic_create_inputs": [],
        "diagnostic_discarded": [],
        "diagnostic_sandboxes": {},
        "applied": [],
        "promoted": [],
        "discarded": [],
        "resolved": [],
        "render_inputs": [],
        "saturated_gens": [],
    }
    remaining = list(pf_results)
    remaining_resolved = list(resolved_values or [])

    def fake_load_full_net(_scenario: Any) -> Any:
        return initial_working

    def fake_load_card(_scenario: Any) -> str:
        return "frozen card"

    def fake_run_ac_pf(net: Any) -> PowerFlowResult:
        state["run_inputs"].append(net)
        return remaining.pop(0)

    def fake_create_sandbox(working: Any, scenario_request_id: Any = None) -> str:
        if scenario_request_id is None:
            state["diagnostic_create_inputs"].append(working)
            sandbox = f"diagnostic-sandbox-{len(state['diagnostic_create_inputs'])}"
            state["diagnostic_sandboxes"][sandbox] = copy.deepcopy(working)
            return sandbox
        state["create_inputs"].append(working)
        return f"sandbox-{len(state['create_inputs'])}"

    def fake_apply_maneuver(
        sandbox: Any,
        maneuver: Maneuver,
        *,
        saturated_gens: frozenset[int] = frozenset(),
    ) -> Any:
        state["applied"].append((sandbox, maneuver))
        state["saturated_gens"].append(saturated_gens)
        if apply_hook is not None:
            apply_hook(sandbox, maneuver)
        return sandbox

    def fake_promote_sandbox(sandbox: Any) -> None:
        state["promoted"].append(sandbox)

    def fake_discard_sandbox(sandbox: Any) -> None:
        if sandbox in state["diagnostic_sandboxes"]:
            state["diagnostic_discarded"].append(sandbox)
            del state["diagnostic_sandboxes"][sandbox]
            return
        state["discarded"].append(sandbox)

    def fake_resolve_net(sandbox: Any) -> Any:
        if sandbox in state["diagnostic_sandboxes"]:
            return state["diagnostic_sandboxes"][sandbox]
        resolved = remaining_resolved.pop(0) if remaining_resolved else f"net{len(state['resolved']) + 1}"
        state["resolved"].append((sandbox, resolved))
        return resolved

    def fake_render_scenario_card(net: Any) -> str:
        state["render_inputs"].append(copy.deepcopy(net))
        if render_card is not None:
            return render_card(net)
        return f"rendered card for {net}"

    monkeypatch.setattr(orch, "load_full_net", fake_load_full_net)
    monkeypatch.setattr(orch, "load_card", fake_load_card)
    monkeypatch.setattr(orch, "run_ac_pf", fake_run_ac_pf)
    monkeypatch.setattr(orch, "create_sandbox", fake_create_sandbox)
    monkeypatch.setattr(orch, "apply_maneuver", fake_apply_maneuver)
    monkeypatch.setattr(orch, "promote_sandbox", fake_promote_sandbox)
    monkeypatch.setattr(orch, "discard_sandbox", fake_discard_sandbox)
    monkeypatch.setattr(orch, "resolve_net", fake_resolve_net)
    monkeypatch.setattr(orch, "render_scenario_card", fake_render_scenario_card, raising=False)
    return state


@pytest.mark.slow
def test_convergent_maneuver_is_not_success_when_slack_p_is_infeasible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Real physics, no fakes: S0166's own witness resolves it to a converged state that draws
    # ~705 MW from the slack. No maneuver in the action space pushes the real 805.2 MW ceiling,
    # so the declared bound is tightened below the resolved draw — the state the solver reaches
    # is genuine, and the run must land on SOLVED_INFEASIBLE rather than SUCCESS.
    net = pp.from_json(str(DATASET_DIR / "full" / "S0166.json"))
    net.ext_grid.at[0, "max_p_mw"] = 690.0
    full_path = tmp_path / "S0166.json"
    pp.to_json(net, str(full_path))
    card_path = tmp_path / "S0166.md"
    card_path.write_text(
        (DATASET_DIR / "llm" / "S0166.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    scenario = Scenario(
        scenario_id="S0166",
        dataset_version="ieee118-reactive-deficit-v1",
        full_net_path=str(full_path),
        card_path=str(card_path),
        memory_split="held_out",
    )
    promoted: list[Any] = []
    original_promote = sandbox_tools.promote_sandbox

    def spy_promote(sandbox: Any) -> None:
        promoted.append(sandbox)
        original_promote(sandbox)

    monkeypatch.setattr(orch, "promote_sandbox", spy_promote)

    def agent_step(**_kwargs: Any) -> orch.AgentStepResult:
        return orch.AgentStepResult(
            maneuver=_maneuver(gen_id=0, vm_pu=0.965),
            llm_responses=(_llm_response(),),
        )

    try:
        response = orch.resolve(
            scenario,
            _config(budget=1, max_runtime_seconds=120),
            agent_step,
        )
    finally:
        for sandbox in promoted:
            sandbox_tools.discard_sandbox(sandbox)

    assert response.status == "BUDGET_EXHAUSTED"
    assert response.converged is False
    assert response.n_maneuvers == 1
    assert response.quality is None
    assert [failure.kind for failure in response.failure_feedback] == [
        "SOLVED_INFEASIBLE"
    ]
    assert "external-grid P=" in (
        response.failure_feedback[0].detail or ""
    )
    assert promoted
    assert response.trace.n_power_flows == 2
    assert response.trace.n_tool_calls == 3
    assert response.trace.n_llm_calls == 1


def test_never_converging_agent_exhausts_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_environment(monkeypatch, [_pf_diverged(1), _pf_diverged(2), _pf_diverged(3), _pf_diverged(4)])

    def agent_step(**_kwargs: Any) -> orch.AgentStepResult:
        return orch.AgentStepResult(maneuver=_maneuver(), llm_responses=(_llm_response(),))

    response = orch.resolve(load_scenario("S0008"), _config(budget=3), agent_step)

    assert response.status == "BUDGET_EXHAUSTED"
    assert response.converged is False
    assert response.n_maneuvers == 3
    assert response.quality is None


def test_converged_non_voltage_infeasible_state_is_not_iterative_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_environment(
        monkeypatch,
        [_pf_diverged(1), _pf_converged(non_voltage_ok=False)],
    )

    def agent_step(**_kwargs: Any) -> orch.AgentStepResult:
        return orch.AgentStepResult(maneuver=_maneuver())

    response = orch.resolve(
        load_scenario("S0008"),
        _config(budget=1),
        agent_step,
    )

    assert response.status == "BUDGET_EXHAUSTED"
    assert response.converged is False
    assert [failure.kind for failure in response.failure_feedback] == [
        "SOLVED_INFEASIBLE"
    ]


def test_converged_voltage_poor_state_remains_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_environment(monkeypatch, [_pf_diverged(1), _pf_converged()])

    def agent_step(**_kwargs: Any) -> orch.AgentStepResult:
        return orch.AgentStepResult(maneuver=_maneuver())

    response = orch.resolve(
        load_scenario("S0008"),
        _config(budget=1),
        agent_step,
    )

    assert response.status == "SUCCESS"
    assert response.converged is True


def test_target_response_carries_canonical_version_stamp(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_environment(monkeypatch, [_pf_diverged(1), _pf_diverged(2)])
    scenario = load_scenario("S0008").model_copy(
        update={"dataset_version": "reactive-deficit-v1"}
    )

    def agent_step(**_kwargs: Any) -> orch.AgentStepResult:
        return orch.AgentStepResult(maneuver=_maneuver())

    response = orch.resolve(scenario, _config(budget=1), agent_step)

    assert response.dataset_version == "reactive-deficit-v1"
    assert response.solver_version == SOLVER_PROBE_POLICY_VERSION
    assert response.action_policy_version == ACTION_POLICY_VERSION
    assert response.ranking_policy_version == RANKING_POLICY_VERSION
    assert response.result_schema_version == RESULT_SCHEMA_VERSION


def test_invalid_action_burns_slot_discards_sandbox_and_keeps_previous_working_net(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"apply": 0}

    def apply_hook(_sandbox: Any, maneuver: Maneuver) -> None:
        calls["apply"] += 1
        if calls["apply"] == 1:
            raise InvalidActionError(maneuver.action, "out of bounds")

    state = _install_fake_environment(monkeypatch, [_pf_diverged(1), _pf_converged()], apply_hook=apply_hook)
    maneuvers = [_maneuver(vm_pu=1.0), _maneuver()]

    def agent_step(**_kwargs: Any) -> orch.AgentStepResult:
        return orch.AgentStepResult(maneuver=maneuvers.pop(0), llm_responses=(_llm_response(),))

    response = orch.resolve(load_scenario("S0008"), _config(budget=2), agent_step)

    assert response.status == "SUCCESS"
    assert [failure.kind for failure in response.failure_feedback] == ["INVALID_ACTION"]
    # sandbox-1: discarded on the invalid action; sandbox-2: discarded after SUCCESS
    # (accepted handles are dropped once the net is resolved — registry hygiene).
    assert state["discarded"] == ["sandbox-1", "sandbox-2"]
    assert state["create_inputs"] == ["net0", "net0"]


def test_repeated_failed_action_is_blocked_while_grid_state_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"apply": 0}

    def apply_hook(_sandbox: Any, maneuver: Maneuver) -> None:
        calls["apply"] += 1
        raise InvalidActionError(maneuver.action, "out of bounds")

    state = _install_fake_environment(monkeypatch, [_pf_diverged(1)], apply_hook=apply_hook)
    first = _maneuver(vm_pu=1.04)
    repeated_with_new_explanation = first.model_copy(
        update={
            "diagnosed_cause": "BAD_SETPOINTS",
            "rationale": "Same physical action, different explanation.",
        }
    )
    proposals = [first, repeated_with_new_explanation]

    def agent_step(**_kwargs: Any) -> orch.AgentStepResult:
        return orch.AgentStepResult(maneuver=proposals.pop(0), llm_responses=(_llm_response(),))

    response = orch.resolve(load_scenario("S0008"), _config(budget=2), agent_step)

    assert response.status == "BUDGET_EXHAUSTED"
    assert calls["apply"] == 1
    assert state["create_inputs"] == ["net0"]
    assert [failure.kind for failure in response.failure_feedback] == ["INVALID_ACTION", "INVALID_ACTION"]
    assert "already failed on the current grid state" in (response.failure_feedback[-1].detail or "")


def test_failed_action_registry_resets_after_applied_maneuver_changes_grid_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_environment(monkeypatch, [_pf_diverged(1), _pf_diverged(2), _pf_converged()])
    repeated = _maneuver(vm_pu=1.04)
    seen_active_failures: list[list[str]] = []

    def agent_step(*, failures: list[Any], **_kwargs: Any) -> orch.AgentStepResult:
        seen_active_failures.append([failure.kind for failure in failures])
        return orch.AgentStepResult(maneuver=repeated, llm_responses=(_llm_response(),))

    response = orch.resolve(load_scenario("S0008"), _config(budget=2), agent_step)

    assert response.status == "SUCCESS"
    assert len(state["applied"]) == 2
    assert seen_active_failures == [[], []]
    assert [failure.kind for failure in response.failure_feedback] == ["STILL_DIVERGED"]


def test_failed_tool_preview_is_persisted_and_blocks_same_final_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_environment(monkeypatch, [_pf_diverged(1), _pf_converged()])
    failed_preview = _maneuver(vm_pu=1.04)

    def agent_step(**_kwargs: Any) -> orch.AgentStepResult:
        return orch.AgentStepResult(
            maneuver=failed_preview,
            llm_responses=(_llm_response(),),
            failed_attempts=(
                FailedManeuverAttempt(
                    kind="PREVIEW_DIVERGED",
                    maneuver=failed_preview,
                    diagnostics=_diagnostics(1),
                    detail="preview did not converge",
                ),
            ),
        )

    response = orch.resolve(load_scenario("S0008"), _config(budget=1), agent_step)

    assert response.status == "BUDGET_EXHAUSTED"
    assert state["applied"] == []
    assert [failure.kind for failure in response.failure_feedback] == ["PREVIEW_DIVERGED", "INVALID_ACTION"]


def test_failed_tool_previews_reset_after_a_different_action_advances_grid_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_environment(monkeypatch, [_pf_diverged(1), _pf_diverged(2), _pf_converged()])
    failed_preview = _maneuver(gen_id=11, vm_pu=1.04)
    advancing_maneuver = _maneuver(gen_id=12, vm_pu=1.04)
    iteration = {"value": 0}
    seen_active_failures: list[list[str]] = []

    def agent_step(*, failures: list[Any], **_kwargs: Any) -> orch.AgentStepResult:
        seen_active_failures.append([failure.kind for failure in failures])
        if iteration["value"] == 0:
            iteration["value"] += 1
            return orch.AgentStepResult(
                maneuver=advancing_maneuver,
                llm_responses=(_llm_response(),),
                failed_attempts=(
                    FailedManeuverAttempt(
                        kind="PREVIEW_DIVERGED",
                        maneuver=failed_preview,
                        diagnostics=_diagnostics(1),
                        detail="preview did not converge",
                    ),
                ),
            )
        return orch.AgentStepResult(maneuver=failed_preview, llm_responses=(_llm_response(),))

    response = orch.resolve(load_scenario("S0008"), _config(budget=2), agent_step)

    assert response.status == "SUCCESS"
    assert [maneuver for _sandbox, maneuver in state["applied"]] == [advancing_maneuver, failed_preview]
    assert seen_active_failures == [[], []]
    assert [failure.kind for failure in response.failure_feedback] == ["PREVIEW_DIVERGED", "STILL_DIVERGED"]


def test_configuration_2_agent_receives_discarded_diagnostic_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_fake_environment(monkeypatch, [_pf_diverged(1), _pf_converged()])
    seen_working_nets: list[Any] = []

    def agent_step(*, grid: Any, **_kwargs: Any) -> orch.AgentStepResult:
        assert grid is not None
        seen_working_nets.append(orch.resolve_net(grid))
        return orch.AgentStepResult(maneuver=_maneuver(), llm_responses=(_llm_response(),))

    response = orch.resolve(load_scenario("S0008"), _config(configuration=2, budget=1), agent_step)

    assert response.status == "SUCCESS"
    assert seen_working_nets == ["net0"]
    assert state["diagnostic_create_inputs"] == ["net0"]
    assert state["diagnostic_discarded"] == ["diagnostic-sandbox-1"]
    assert state["diagnostic_sandboxes"] == {}


@pytest.mark.parametrize("configuration", [3])
def test_tool_configurations_receive_fresh_discarded_grid_copy(
    monkeypatch: pytest.MonkeyPatch,
    configuration: int,
) -> None:
    state = _install_fake_environment(monkeypatch, [_pf_diverged(1), _pf_diverged(2), _pf_converged()])
    seen_working_nets: list[Any] = []

    def agent_step(*, grid: Any, **_kwargs: Any) -> orch.AgentStepResult:
        assert grid is not None
        seen_working_nets.append(orch.resolve_net(grid))
        return orch.AgentStepResult(maneuver=_maneuver(), llm_responses=(_llm_response(),))

    response = orch.resolve(load_scenario("S0008"), _config(configuration=configuration, budget=2), agent_step)

    assert response.status == "SUCCESS"
    assert seen_working_nets == ["net0", "net1"]
    assert state["diagnostic_create_inputs"] == ["net0", "net1"]
    assert state["diagnostic_discarded"] == ["diagnostic-sandbox-1", "diagnostic-sandbox-2"]
    assert state["diagnostic_sandboxes"] == {}


def test_mutating_diagnostic_grid_does_not_mutate_working_net(monkeypatch: pytest.MonkeyPatch) -> None:
    initial_working = {"gen_vm_pu": 1.0}
    state = _install_fake_environment(
        monkeypatch,
        [_pf_diverged(1), _pf_converged()],
        initial_working=initial_working,
    )

    def agent_step(*, grid: Any, **_kwargs: Any) -> orch.AgentStepResult:
        assert grid is not None
        orch.resolve_net(grid)["gen_vm_pu"] = 0.95
        return orch.AgentStepResult(maneuver=_maneuver(), llm_responses=(_llm_response(),))

    response = orch.resolve(load_scenario("S0008"), _config(configuration=3, budget=1), agent_step)

    assert response.status == "SUCCESS"
    assert state["create_inputs"] == [{"gen_vm_pu": 1.0}]
    assert initial_working == {"gen_vm_pu": 1.0}
    assert state["diagnostic_sandboxes"] == {}


def test_non_maneuver_result_is_malformed_output_and_burns_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_environment(monkeypatch, [_pf_diverged(1), _pf_converged()])
    calls = {"agent": 0}

    def agent_step(**_kwargs: Any) -> orch.AgentStepResult:
        calls["agent"] += 1
        if calls["agent"] == 1:
            return orch.AgentStepResult(maneuver={"not": "a maneuver"})  # type: ignore[arg-type]
        return orch.AgentStepResult(maneuver=_maneuver(), llm_responses=(_llm_response(),))

    response = orch.resolve(load_scenario("S0008"), _config(budget=2), agent_step)

    assert response.status == "SUCCESS"
    assert [failure.kind for failure in response.failure_feedback] == ["MALFORMED_OUTPUT"]
    assert response.n_maneuvers == 1


def test_still_diverged_maneuver_advances_state_and_fresh_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_environment(monkeypatch, [_pf_diverged(1), _pf_diverged(2), _pf_converged()])
    seen_buses: list[int | None] = []

    def agent_step(*, diagnostics: NRDiagnostics | None, **_kwargs: Any) -> orch.AgentStepResult:
        seen_buses.append(None if diagnostics is None else diagnostics.lowest_vm_bus)
        return orch.AgentStepResult(maneuver=_maneuver(), llm_responses=(_llm_response(),))

    response = orch.resolve(load_scenario("S0008"), _config(budget=2), agent_step)

    assert response.status == "SUCCESS"
    assert seen_buses == [1, 2]
    assert state["create_inputs"] == ["net0", "net1"]
    assert [failure.kind for failure in response.failure_feedback] == ["STILL_DIVERGED"]


def test_agent_loop_rerenders_card_from_current_working_net_each_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_environment(
        monkeypatch,
        [_pf_diverged(1), _pf_diverged(2), _pf_converged()],
        initial_working={"gen_vm_pu": 1.0},
        resolved_values=[{"gen_vm_pu": 1.05}],
        render_card=lambda net: f"live card gen_vm_pu={net['gen_vm_pu']}",
    )
    seen_cards: list[str] = []

    def agent_step(*, card: str, **_kwargs: Any) -> orch.AgentStepResult:
        seen_cards.append(card)
        return orch.AgentStepResult(maneuver=_maneuver(), llm_responses=(_llm_response(),))

    response = orch.resolve(load_scenario("S0008"), _config(configuration=2, budget=2), agent_step)

    assert response.status == "SUCCESS"
    assert seen_cards == ["live card gen_vm_pu=1.0", "live card gen_vm_pu=1.05"]
    assert state["render_inputs"] == [{"gen_vm_pu": 1.0}, {"gen_vm_pu": 1.05}]


def test_timeout_preserves_applied_maneuvers(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_environment(monkeypatch, [_pf_diverged(1), _pf_diverged(2)])
    calls = {"agent": 0}

    def fake_monotonic() -> float:
        return 20.0 if calls["agent"] else 0.0

    monkeypatch.setattr(orch.time, "monotonic", fake_monotonic)

    def agent_step(**_kwargs: Any) -> orch.AgentStepResult:
        calls["agent"] += 1
        return orch.AgentStepResult(maneuver=_maneuver(), llm_responses=(_llm_response(),))

    response = orch.resolve(load_scenario("S0008"), _config(budget=2), agent_step)

    assert response.status == "TIMEOUT"
    assert response.converged is False
    assert response.n_maneuvers == 1


def test_llm_failure_ends_run_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_environment(monkeypatch, [_pf_diverged(1)])
    calls = {"agent": 0}

    def agent_step(**_kwargs: Any) -> orch.AgentStepResult:
        calls["agent"] += 1
        raise LLMFailureError("test-model", "bedrock failed")

    response = orch.resolve(load_scenario("S0008"), _config(), agent_step)

    assert response.status == "LLM_FAILURE"
    assert response.n_maneuvers == 0
    assert calls["agent"] == 1


def test_llm_failure_still_records_the_tokens_the_agent_already_spent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run that dies mid-agent still billed AWS for every call it made; the trace must count them.

    Recording zero tokens on failure understates the cost of exactly the configurations that fail
    most, which corrupts the cost comparison.
    """
    from restorebench.schemas.errors import attach_llm_responses

    _install_fake_environment(monkeypatch, [_pf_diverged(1)])

    def agent_step(**_kwargs: Any) -> orch.AgentStepResult:
        error = LLMFailureError("test-model", "bedrock failed")
        raise attach_llm_responses(error, (_llm_response(40, 12), _llm_response(30, 8)))

    response = orch.resolve(load_scenario("S0008"), _config(), agent_step)

    assert response.status == "LLM_FAILURE"
    assert response.trace.total_llm_tokens_in == 70
    assert response.trace.total_llm_tokens_out == 20
    assert response.trace.n_llm_calls == 2


def test_malformed_output_still_records_the_tokens_the_agent_already_spent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic import ValidationError

    from restorebench.schemas.errors import attach_llm_responses

    _install_fake_environment(monkeypatch, [_pf_diverged(1), _pf_diverged(2)])

    def agent_step(**_kwargs: Any) -> orch.AgentStepResult:
        try:
            Maneuver.model_validate({})
        except ValidationError as exc:
            raise attach_llm_responses(exc, (_llm_response(50, 15),)) from exc
        raise AssertionError("unreachable")

    response = orch.resolve(load_scenario("S0008"), _config(budget=1), agent_step)

    assert response.trace.total_llm_tokens_in == 50
    assert response.trace.total_llm_tokens_out == 15
    assert response.trace.n_llm_calls == 1


def test_converging_scenario_on_load_raises_corpus_integrity(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_environment(monkeypatch, [_pf_converged()])

    def agent_step(**_kwargs: Any) -> orch.AgentStepResult:
        return orch.AgentStepResult(maneuver=_maneuver())

    with pytest.raises(CorpusIntegrityError):
        orch.resolve(load_scenario("S0008"), _config(), agent_step)


def test_reasoning_entries_capture_role_iteration_and_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_environment(monkeypatch, [_pf_diverged(1), _pf_converged()])

    def agent_step(**_kwargs: Any) -> orch.AgentStepResult:
        return orch.AgentStepResult(
            maneuver=_maneuver(),
            llm_responses=(
                _llm_response(raw={"role": "analyst", "reasoning": "find weak bus"}),
                _llm_response(raw={"role": "executor", "reasoning": "raise generator voltage"}),
            ),
        )

    response = orch.resolve(load_scenario("S0008"), _config(), agent_step)

    assert [entry.model_dump() for entry in response.trace.reasoning] == [
        {"iteration": 0, "role": "analyst", "text": "find weak bus"},
        {"iteration": 0, "role": "executor", "text": "raise generator voltage"},
    ]


def test_reasoning_defaults_role_and_ignores_missing_or_empty_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_environment(monkeypatch, [_pf_diverged(1), _pf_converged()])

    def agent_step(**_kwargs: Any) -> orch.AgentStepResult:
        return orch.AgentStepResult(
            maneuver=_maneuver(),
            llm_responses=(
                _llm_response(raw={"reasoning": "default role path"}),
                _llm_response(raw={}),
                _llm_response(raw={"role": "analyst", "reasoning": ""}),
            ),
        )

    response = orch.resolve(load_scenario("S0008"), _config(), agent_step)

    assert [entry.model_dump() for entry in response.trace.reasoning] == [
        {"iteration": 0, "role": "agent", "text": "default role path"}
    ]
    assert response.trace.n_llm_calls == 3


def test_reasoning_entries_preserve_multi_iteration_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_environment(monkeypatch, [_pf_diverged(1), _pf_diverged(2), _pf_diverged(3), _pf_converged()])
    calls = {"agent": 0}

    def agent_step(**_kwargs: Any) -> orch.AgentStepResult:
        iteration = calls["agent"]
        calls["agent"] += 1
        return orch.AgentStepResult(
            maneuver=_maneuver(),
            llm_responses=(_llm_response(raw={"role": "single_agent", "reasoning": f"step {iteration}"}),),
        )

    response = orch.resolve(load_scenario("S0008"), _config(budget=3), agent_step)

    assert [(entry.iteration, entry.role, entry.text) for entry in response.trace.reasoning] == [
        (0, "single_agent", "step 0"),
        (1, "single_agent", "step 1"),
        (2, "single_agent", "step 2"),
    ]


def test_case1_sequence_stops_at_first_converging_maneuver(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_environment(monkeypatch, [_pf_diverged(1), _pf_diverged(2), _pf_diverged(3), _pf_converged()])

    def propose_sequence(**_kwargs: Any) -> orch.ProposeSequenceResult:
        return orch.ProposeSequenceResult(
            maneuvers=ManeuverSequence(
                maneuvers=[_maneuver(11), _maneuver(12), _maneuver(13), _maneuver(14), _maneuver(15)],
                reconstruction_summary="candidate sequence",
            ),
            llm_responses=(_llm_response(raw={"role": "chatbot", "reasoning": "try a sequence"}),),
        )

    response = orch.resolve(load_scenario("S0008"), _config(configuration=1, budget=5), propose_sequence)

    assert response.status == "SUCCESS"
    assert response.n_maneuvers == 3
    assert len(response.maneuvers) == 3
    assert [entry.model_dump() for entry in response.trace.reasoning] == [
        {"iteration": 0, "role": "chatbot", "text": "try a sequence"}
    ]


def test_case1_passes_baseline_q_saturation_to_authoritative_applier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_environment(monkeypatch, [_pf_diverged(1), _pf_converged()])

    def propose_sequence(**_kwargs: Any) -> orch.ProposeSequenceResult:
        return orch.ProposeSequenceResult(
            maneuvers=ManeuverSequence(
                maneuvers=[_maneuver(11)],
                reconstruction_summary="candidate sequence",
            ),
            llm_responses=(_llm_response(),),
        )

    response = orch.resolve(load_scenario("S0008"), _config(configuration=1, budget=1), propose_sequence)

    assert response.status == "SUCCESS"
    assert state["saturated_gens"] == [frozenset({1, 2})]


def test_case1_blocks_repeated_failed_action_while_grid_state_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"apply": 0}

    def apply_hook(_sandbox: Any, maneuver: Maneuver) -> None:
        calls["apply"] += 1
        raise InvalidActionError(maneuver.action, "out of bounds")

    state = _install_fake_environment(monkeypatch, [_pf_diverged(1)], apply_hook=apply_hook)
    first = _maneuver(vm_pu=1.04)
    repeated = first.model_copy(update={"rationale": "Same action repeated in the proposed sequence."})

    def propose_sequence(**_kwargs: Any) -> orch.ProposeSequenceResult:
        return orch.ProposeSequenceResult(
            maneuvers=ManeuverSequence(
                maneuvers=[first, repeated],
                reconstruction_summary="candidate sequence",
            ),
            llm_responses=(_llm_response(),),
        )

    response = orch.resolve(load_scenario("S0008"), _config(configuration=1, budget=2), propose_sequence)

    assert response.status == "BUDGET_EXHAUSTED"
    assert calls["apply"] == 1
    assert state["create_inputs"] == ["net0"]
    assert [failure.kind for failure in response.failure_feedback] == ["INVALID_ACTION", "INVALID_ACTION"]
    assert "already failed on the current grid state" in (response.failure_feedback[-1].detail or "")


def test_case1_built_chatbot_runs_through_existing_applier_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_environment(monkeypatch, [_pf_diverged(1), _pf_diverged(2), _pf_converged()])

    def fake_llm_call(
        _model_id: str,
        _messages: list[ChatMessage],
        **_kwargs: Any,
    ) -> LLMResponse:
        return _sequence_llm_response([_maneuver(11), _maneuver(12), _maneuver(13)])

    monkeypatch.setattr(baseline_chatbot, "llm_call", fake_llm_call)

    response = orch.resolve(
        load_scenario("S0008"),
        _config(configuration=1, budget=3),
        baseline_chatbot.make_baseline_chatbot("test-model"),
    )

    assert response.status == "SUCCESS"
    assert response.n_maneuvers == 2
    assert response.maneuvers == [_maneuver(11), _maneuver(12)]
    assert [(entry.iteration, entry.role) for entry in response.trace.reasoning] == [(0, "chatbot")]


def test_case1_converged_non_voltage_infeasible_state_is_not_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_environment(
        monkeypatch,
        [_pf_diverged(1), _pf_converged(non_voltage_ok=False)],
    )

    def propose_sequence(**_kwargs: Any) -> orch.ProposeSequenceResult:
        return orch.ProposeSequenceResult(
            maneuvers=ManeuverSequence(
                maneuvers=[_maneuver()],
                reconstruction_summary="one constrained attempt",
            )
        )

    response = orch.resolve(
        load_scenario("S0008"),
        _config(configuration=1, budget=1),
        propose_sequence,
    )

    assert response.status == "BUDGET_EXHAUSTED"
    assert [failure.kind for failure in response.failure_feedback] == [
        "SOLVED_INFEASIBLE"
    ]


def test_case1_over_budget_sequence_is_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_fake_environment(monkeypatch, [_pf_diverged(1), _pf_diverged(2), _pf_diverged(3)])

    def propose_sequence(**_kwargs: Any) -> orch.ProposeSequenceResult:
        return orch.ProposeSequenceResult(
            maneuvers=ManeuverSequence(
                maneuvers=[_maneuver(11), _maneuver(12), _maneuver(13), _maneuver(14)],
                reconstruction_summary="candidate sequence",
            ),
            llm_responses=(_llm_response(raw={"role": "chatbot", "reasoning": "try a long sequence"}),),
        )

    response = orch.resolve(load_scenario("S0008"), _config(configuration=1, budget=2), propose_sequence)

    assert response.status == "BUDGET_EXHAUSTED"
    assert response.n_maneuvers == 2
    assert [maneuver for _sandbox, maneuver in state["applied"]] == [_maneuver(11), _maneuver(12)]


def test_case1_empty_sequence_exhausts_budget_without_crashing(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_environment(monkeypatch, [_pf_diverged(1)])

    def propose_sequence(**_kwargs: Any) -> orch.ProposeSequenceResult:
        return orch.ProposeSequenceResult(
            maneuvers=ManeuverSequence(maneuvers=[], reconstruction_summary=None),
            llm_responses=(_llm_response(),),
        )

    response = orch.resolve(load_scenario("S0008"), _config(configuration=1), propose_sequence)

    assert response.status == "BUDGET_EXHAUSTED"
    assert response.n_maneuvers == 0
    assert response.failure_feedback == []


def test_configuration3_built_multi_agent_resolves_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_tiny_runner(monkeypatch, [_pf_diverged(1), _pf_converged()])
    _install_multi_agent_llm(
        monkeypatch,
        RoutedLLM(
            {
                "analyst-model": [_multi_analyst_response()],
                "executor-model": [_multi_executor_response()],
                "orchestrator-model": [_multi_decision_response()],
            }
        ),
    )

    response = orch.resolve(
        load_scenario("S0008"), _multi_config(configuration=3, budget=1), multi_agent.make_multi_agent()
    )

    assert response.status == "SUCCESS"
    assert response.n_maneuvers == 1
    assert [entry.role for entry in response.trace.reasoning] == ["analyst", "executor", "orchestrator_agent"]


def test_configuration3_built_multi_agent_continues_after_still_diverged(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_tiny_runner(monkeypatch, [_pf_diverged(1), _pf_diverged(2), _pf_converged()])
    _install_multi_agent_llm(
        monkeypatch,
        RoutedLLM(
            {
                "analyst-model": [_multi_analyst_response(1.01), _multi_analyst_response(1.02)],
                "executor-model": [_multi_executor_response(1.01), _multi_executor_response(1.02)],
                "orchestrator-model": [_multi_decision_response(), _multi_decision_response()],
            }
        ),
    )

    response = orch.resolve(
        load_scenario("S0008"), _multi_config(configuration=3, budget=2), multi_agent.make_multi_agent()
    )

    assert response.status == "SUCCESS"
    assert response.n_maneuvers == 2
    assert [failure.kind for failure in response.failure_feedback] == ["STILL_DIVERGED"]


def test_baseline_diagnostics_event_payload_carries_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_tiny_runner(monkeypatch, [_pf_diverged(1), _pf_converged()])
    promoted: list[Any] = []
    original_promote = sandbox_tools.promote_sandbox

    def tracking_promote(sandbox: Any) -> None:
        promoted.append(sandbox)
        original_promote(sandbox)

    monkeypatch.setattr(orch, "promote_sandbox", tracking_promote)

    def agent_step(**_kwargs: Any) -> orch.AgentStepResult:
        return orch.AgentStepResult(maneuver=_maneuver(), llm_responses=(_llm_response(),))

    try:
        response = orch.resolve(load_scenario("S0008"), _config(max_runtime_seconds=120), agent_step)
    finally:
        for sandbox in promoted:
            sandbox_tools.discard_sandbox(sandbox)

    baseline_events = [event for event in response.trace.events if event.event_name == "baseline_diagnostics"]
    assert len(baseline_events) == 1
    assert NRDiagnostics.model_validate(baseline_events[0].payload["diagnostics"]) == _diagnostics(1)


def test_resolve_leaves_the_sandbox_registry_empty_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    # Accepted sandboxes were promoted but never discarded: each applied maneuver leaked a
    # deepcopied net in the module-global registry for the life of the harness process.
    _install_tiny_runner(monkeypatch, [_pf_diverged(1), _pf_converged()])

    def agent_step(**_kwargs: Any) -> orch.AgentStepResult:
        return orch.AgentStepResult(maneuver=_maneuver(gen_id=0, vm_pu=1.01), llm_responses=(_llm_response(),))

    before = len(sandbox_tools._SANDBOX_REGISTRY)
    response = orch.resolve(load_scenario("S0008"), _config(max_runtime_seconds=120), agent_step)

    assert response.status == "SUCCESS"
    assert len(sandbox_tools._SANDBOX_REGISTRY) == before


def test_resolve_leaves_the_sandbox_registry_empty_after_still_diverged_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tiny_runner(monkeypatch, [_pf_diverged(1), _pf_diverged(1), _pf_diverged(1)])

    def agent_step(**_kwargs: Any) -> orch.AgentStepResult:
        return orch.AgentStepResult(maneuver=_maneuver(gen_id=0, vm_pu=1.01), llm_responses=(_llm_response(),))

    before = len(sandbox_tools._SANDBOX_REGISTRY)
    response = orch.resolve(load_scenario("S0008"), _config(budget=2, max_runtime_seconds=120), agent_step)

    assert response.status == "BUDGET_EXHAUSTED"
    assert len(sandbox_tools._SANDBOX_REGISTRY) == before


def test_case1_resolve_leaves_the_sandbox_registry_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_tiny_runner(monkeypatch, [_pf_diverged(1), _pf_converged()])
    fake_llm = RoutedLLM({"test-model": [_sequence_llm_response([_maneuver(gen_id=0, vm_pu=1.01)])]})
    monkeypatch.setattr(baseline_chatbot, "llm_call", fake_llm)

    before = len(sandbox_tools._SANDBOX_REGISTRY)
    config = OrchestratorConfig(
        CONFIGURATION=1,
        MANEUVER_BUDGET=2,
        MAX_RUNTIME_SECONDS=120,
        LLM_ASSIGNMENT=_assignment(),
    )
    response = orch.resolve(load_scenario("S0008"), config, baseline_chatbot.make_baseline_chatbot("test-model"))

    assert response.status == "SUCCESS"
    assert len(sandbox_tools._SANDBOX_REGISTRY) == before


def test_trace_aggregates_total_tokens_across_llm_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_tiny_runner(monkeypatch, [_pf_diverged(1), _pf_converged()])

    def agent_step(**_kwargs: Any) -> orch.AgentStepResult:
        first = _llm_response(tokens_in=100, tokens_out=40)
        first.tokens_total = 150  # Bedrock reported more than in+out (cache tokens)
        second = _llm_response(tokens_in=200, tokens_out=60)
        second.tokens_total = 260
        return orch.AgentStepResult(maneuver=_maneuver(gen_id=0, vm_pu=1.01), llm_responses=(first, second))

    response = orch.resolve(load_scenario("S0008"), _config(max_runtime_seconds=120), agent_step)

    assert response.trace.total_llm_tokens_in == 300
    assert response.trace.total_llm_tokens_out == 100
    assert response.trace.total_llm_tokens == 410  # what Bedrock billed, not the in+out sum


def _pf_diverged_with_overstress(bus: int, overstress: float) -> PowerFlowResult:
    diagnostics = _diagnostics(bus).model_copy(update={"overstress": overstress})
    return PowerFlowResult(
        converged=False,
        iterations=30,
        tolerance_used=1e-6,
        runtime_ms=1.0,
        error_message="did not converge",
        diagnostics=diagnostics,
    )


def test_accepted_history_carries_the_overstress_each_maneuver_produced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The agent restarts its conversation each iteration, so progress has to travel in the history."""
    _install_fake_environment(
        monkeypatch,
        [
            _pf_diverged_with_overstress(1, 0.0333),
            _pf_diverged_with_overstress(2, 0.0281),
            _pf_diverged_with_overstress(3, 0.0244),
        ],
    )
    seen_histories: list[Any] = []
    proposals = [_maneuver(vm_pu=1.01), _maneuver(vm_pu=1.02)]

    def agent_step(**kwargs: Any) -> orch.AgentStepResult:
        seen_histories.append(list(kwargs["history"]))
        return orch.AgentStepResult(maneuver=proposals.pop(0), llm_responses=(_llm_response(),))

    response = orch.resolve(load_scenario("S0008"), _config(budget=2), agent_step)

    assert response.status == "BUDGET_EXHAUSTED"
    # The response keeps plain maneuvers; only the agent-facing history gains the outcomes.
    assert [type(item).__name__ for item in response.maneuvers] == ["Maneuver", "Maneuver"]

    assert seen_histories[0] == []
    first, = seen_histories[1]
    assert first.maneuver == response.maneuvers[0]
    assert first.overstress_before == pytest.approx(0.0333)
    assert first.overstress_after == pytest.approx(0.0281)
