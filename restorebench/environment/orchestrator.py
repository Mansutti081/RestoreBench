# ABOUTME: Runs the deterministic diagnose/maneuver/solve loop for one scenario.
# ABOUTME: Owns convergence, budget, trace accounting, and ResolutionResponse construction.
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Protocol, cast
from uuid import UUID, uuid4

from pydantic import ValidationError

from restorebench.environment.card_render import render_scenario_card
from restorebench.environment.scenarios import load_card, load_full_net
from restorebench.llm.providers import LLMResponse
from restorebench.physics.actions import ACTION_POLICY_VERSION
from restorebench.physics.feasibility import satisfies_non_voltage_constraints
from restorebench.physics.policies import RANKING_POLICY_VERSION, SOLVER_PROBE_POLICY_VERSION
from restorebench.schemas.actions import Maneuver, ManeuverSequence
from restorebench.schemas.config import OrchestratorConfig
from restorebench.schemas.dataset import Scenario
from restorebench.schemas.errors import CorpusIntegrityError, InvalidActionError, LLMFailureError, ToolFailureError
from restorebench.schemas.errors import llm_responses_of
from restorebench.schemas.feedback import AcceptedManeuver, FailedManeuverAttempt, FailureFeedback, SandboxNet
from restorebench.schemas.power_flow import NRDiagnostics, PowerFlowResult
from restorebench.schemas.response import (
    RESULT_SCHEMA_VERSION,
    ExecutionTrace,
    ReasoningEntry,
    ResolutionResponse,
    TraceEvent,
)
from restorebench.tools.power_flow import run_ac_pf
from restorebench.tools.sandbox import apply_maneuver, create_sandbox, discard_sandbox, promote_sandbox, resolve_net


Status = Literal["SUCCESS", "BUDGET_EXHAUSTED", "TIMEOUT", "LLM_FAILURE", "TOOL_FAILURE"]
Phase = Literal["baseline", "diagnosis", "maneuver", "solve", "response"]


@dataclass(frozen=True)
class AgentStepResult:
    maneuver: Maneuver
    llm_responses: tuple[LLMResponse, ...] = ()
    failed_attempts: tuple[FailedManeuverAttempt, ...] = ()


@dataclass(frozen=True)
class ProposeSequenceResult:
    maneuvers: ManeuverSequence
    llm_responses: tuple[LLMResponse, ...] = ()


@dataclass(frozen=True)
class AgentCallOutcome:
    step_result: AgentStepResult | None
    status: Literal["OK", "MALFORMED_OUTPUT", "LLM_FAILURE"]
    detail: str | None = None


@dataclass(frozen=True)
class ApplyOutcome:
    sandbox: Any | None
    result: PowerFlowResult | None
    failure: FailureFeedback | None = None
    tool_failure: bool = False


class AgentStep(Protocol):
    def __call__(
        self,
        *,
        card: str,
        grid: SandboxNet | None,
        diagnostics: NRDiagnostics | None,
        history: Sequence[AcceptedManeuver],
        failures: Sequence[FailureFeedback],
        config: OrchestratorConfig,
    ) -> AgentStepResult: ...


class ProposeSequence(Protocol):
    def __call__(self, *, card: str, config: OrchestratorConfig) -> ProposeSequenceResult: ...


class TraceRecorder:
    def __init__(self, request_id: UUID) -> None:
        self.request_id = request_id
        self.events: list[TraceEvent] = []
        self.n_llm_calls = 0
        self.total_llm_tokens_in = 0
        self.total_llm_tokens_out = 0
        self.total_llm_tokens = 0
        self.n_tool_calls = 0
        self.n_power_flows = 0
        self.reasoning: list[ReasoningEntry] = []

    def record(self, phase: Phase, event_name: str, payload: dict[str, Any], started: float) -> None:
        self.events.append(
            TraceEvent(
                timestamp=datetime.now(timezone.utc),
                phase=phase,
                event_name=event_name,
                duration_ms=_elapsed_ms(started),
                payload=payload,
            )
        )

    def add_llm_responses(self, responses: Sequence[LLMResponse], iteration: int) -> None:
        self.n_llm_calls += len(responses)
        self.total_llm_tokens_in += sum(response.tokens_in for response in responses)
        self.total_llm_tokens_out += sum(response.tokens_out for response in responses)
        self.total_llm_tokens += sum(response.tokens_total for response in responses)
        for response in responses:
            reasoning = response.raw.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                self.reasoning.append(
                    ReasoningEntry(
                        iteration=iteration,
                        role=response.raw.get("role", "agent"),
                        text=reasoning,
                    )
                )

    def build(self) -> ExecutionTrace:
        return ExecutionTrace(
            request_id=self.request_id,
            events=self.events,
            n_llm_calls=self.n_llm_calls,
            total_llm_tokens_in=self.total_llm_tokens_in,
            total_llm_tokens_out=self.total_llm_tokens_out,
            total_llm_tokens=self.total_llm_tokens,
            n_tool_calls=self.n_tool_calls,
            n_power_flows=self.n_power_flows,
            reasoning=self.reasoning,
        )


def resolve(
    scenario: Scenario,
    config: OrchestratorConfig,
    agent_step: AgentStep | ProposeSequence,
) -> ResolutionResponse:
    request_id = uuid4()
    started_at = datetime.now(timezone.utc)
    started_runtime = time.monotonic()
    trace = TraceRecorder(request_id)
    working = load_full_net(scenario)
    card = load_card(scenario)

    try:
        baseline = _run_pf(working, trace, "baseline", "baseline_diagnostics", record_diagnostics=True)
    except ToolFailureError:
        return _respond("TOOL_FAILURE", scenario, config, trace, request_id, [], [], started_at, started_runtime)
    if baseline.converged:
        raise CorpusIntegrityError(f"{scenario.scenario_id} converges on load; corpus integrity failed")
    if config.CONFIGURATION == 1:
        proposer = cast(ProposeSequence, agent_step)
        return _resolve_case1(
            scenario,
            config,
            proposer,
            trace,
            request_id,
            started_at,
            started_runtime,
            working,
            card,
            baseline,
        )
    iterative_agent = cast(AgentStep, agent_step)
    return _resolve_agent_loop(
        scenario,
        config,
        iterative_agent,
        trace,
        request_id,
        started_at,
        started_runtime,
        working,
        card,
        baseline,
    )


def _resolve_agent_loop(
    scenario: Scenario,
    config: OrchestratorConfig,
    agent_step: AgentStep,
    trace: TraceRecorder,
    request_id: UUID,
    started_at: datetime,
    started_runtime: float,
    working: Any,
    card: str,
    baseline: PowerFlowResult,
) -> ResolutionResponse:
    accepted: list[AcceptedManeuver] = []
    failures: list[FailureFeedback] = []
    state_failures: list[FailureFeedback] = []
    for iteration in range(config.MANEUVER_BUDGET):
        if _timed_out(started_runtime, config):
            return _respond(
                "TIMEOUT", scenario, config, trace, request_id, _maneuvers_of(accepted), failures, started_at, started_runtime
            )
        card = render_scenario_card(working)
        grid = _agent_grid(config, working)
        try:
            outcome = _call_agent(
                agent_step,
                card,
                grid,
                baseline.diagnostics,
                accepted,
                state_failures,
                config,
                trace,
                iteration,
            )
        finally:
            if grid is not None:
                discard_sandbox(grid)
        if outcome.status == "LLM_FAILURE":
            return _respond(
                "LLM_FAILURE", scenario, config, trace, request_id, _maneuvers_of(accepted), failures, started_at, started_runtime
            )
        if outcome.status == "MALFORMED_OUTPUT" or outcome.step_result is None:
            feedback = _failure(iteration, "MALFORMED_OUTPUT", baseline.diagnostics, outcome.detail, None)
            failures.append(feedback)
            state_failures.append(feedback)
            continue
        step_result = outcome.step_result
        for attempt in step_result.failed_attempts:
            if _action_failed_on_current_state(attempt.maneuver, state_failures):
                continue
            feedback = _failed_attempt_feedback(iteration, attempt)
            failures.append(feedback)
            state_failures.append(feedback)
        if _action_failed_on_current_state(step_result.maneuver, state_failures):
            feedback = _repeated_action_failure(trace, step_result.maneuver, baseline.diagnostics, iteration)
            failures.append(feedback)
            state_failures.append(feedback)
            continue
        applied = _apply_and_solve(trace, request_id, working, step_result.maneuver, iteration, baseline.diagnostics)
        if applied.tool_failure:
            return _respond(
                "TOOL_FAILURE", scenario, config, trace, request_id, _maneuvers_of(accepted), failures, started_at, started_runtime
            )
        if applied.failure is not None:
            failures.append(applied.failure)
            state_failures.append(applied.failure)
            continue
        sandbox, result = applied.sandbox, applied.result
        assert sandbox is not None and result is not None
        accepted.append(
            AcceptedManeuver(
                maneuver=step_result.maneuver,
                overstress_before=_overstress(baseline.diagnostics),
                overstress_after=_overstress(result.diagnostics),
            )
        )
        if _is_successful_terminal(result):
            discard_sandbox(sandbox)
            return _respond(
                "SUCCESS", scenario, config, trace, request_id, _maneuvers_of(accepted), failures, started_at, started_runtime, result
            )
        # resolve_net hands back the applied net object; the registry entry is then dead
        # weight — without the discard every accepted maneuver leaks a deepcopied net in
        # the module-global registry for the life of the harness process.
        working, baseline = resolve_net(sandbox), result
        discard_sandbox(sandbox)
        if result.converged:
            feedback = _failure(
                iteration,
                "SOLVED_INFEASIBLE",
                None,
                _non_voltage_failure_detail(result),
                step_result.maneuver,
            )
            failures.append(feedback)
            state_failures[:] = [feedback]
        else:
            failures.append(
                _failure(iteration, "STILL_DIVERGED", result.diagnostics, result.error_message, step_result.maneuver)
            )
            state_failures.clear()
    return _respond(
        "BUDGET_EXHAUSTED", scenario, config, trace, request_id, _maneuvers_of(accepted), failures, started_at, started_runtime
    )


def _maneuvers_of(accepted: Sequence[AcceptedManeuver]) -> list[Maneuver]:
    return [item.maneuver for item in accepted]


def _overstress(diagnostics: NRDiagnostics | None) -> float | None:
    return None if diagnostics is None else diagnostics.overstress


def _action_failed_on_current_state(
    maneuver: Maneuver,
    state_failures: Sequence[FailureFeedback],
) -> bool:
    return any(
        failure.maneuver is not None and failure.maneuver.action == maneuver.action for failure in state_failures
    )


def _repeated_action_failure(
    trace: TraceRecorder,
    maneuver: Maneuver,
    diagnostics: NRDiagnostics | None,
    iteration: int,
) -> FailureFeedback:
    detail = "the same action already failed on the current grid state; choose a different action"
    started = time.monotonic()
    trace.record(
        "maneuver",
        "repeated_failed_action",
        {"iteration": iteration, "action": maneuver.action.model_dump(mode="json")},
        started,
    )
    return _failure(iteration, "INVALID_ACTION", diagnostics, detail, maneuver)


def _failed_attempt_feedback(
    iteration: int,
    attempt: FailedManeuverAttempt,
) -> FailureFeedback:
    return FailureFeedback(
        iteration=iteration,
        kind=attempt.kind,
        diagnostics=attempt.diagnostics,
        detail=attempt.detail,
        maneuver=attempt.maneuver,
    )


def _resolve_case1(
    scenario: Scenario,
    config: OrchestratorConfig,
    proposer: ProposeSequence,
    trace: TraceRecorder,
    request_id: UUID,
    started_at: datetime,
    started_runtime: float,
    working: Any,
    card: str,
    baseline: PowerFlowResult,
) -> ResolutionResponse:
    maneuvers: list[Maneuver] = []
    failures: list[FailureFeedback] = []
    state_failures: list[FailureFeedback] = []
    try:
        proposal = proposer(card=card, config=config)
    except LLMFailureError as exc:
        trace.add_llm_responses(llm_responses_of(exc), iteration=0)
        # LLM_FAILURE covers an output cap we set, a provider rejecting a request we built, and a
        # model that cannot produce the schema. Discarding the reason makes those indistinguishable
        # in the results, and telling them apart afterwards costs a re-run of the whole scenario.
        failures.append(_failure(0, "LLM_FAILURE", None, str(exc), None))
        return _respond(
            "LLM_FAILURE", scenario, config, trace, request_id, maneuvers, failures, started_at, started_runtime
        )
    except ValidationError as exc:
        trace.add_llm_responses(llm_responses_of(exc), iteration=0)
        failures.append(_failure(0, "MALFORMED_OUTPUT", None, str(exc), None))
        return _respond(
            "BUDGET_EXHAUSTED", scenario, config, trace, request_id, maneuvers, failures, started_at, started_runtime
        )
    trace.add_llm_responses(proposal.llm_responses, iteration=0)
    sequence = proposal.maneuvers
    diagnostics = baseline.diagnostics
    for iteration, maneuver in enumerate(sequence.maneuvers[: config.MANEUVER_BUDGET]):
        if _timed_out(started_runtime, config):
            return _respond(
                "TIMEOUT", scenario, config, trace, request_id, maneuvers, failures, started_at, started_runtime
            )
        if _action_failed_on_current_state(maneuver, state_failures):
            feedback = _repeated_action_failure(trace, maneuver, diagnostics, iteration)
            failures.append(feedback)
            state_failures.append(feedback)
            continue
        applied = _apply_and_solve(trace, request_id, working, maneuver, iteration, diagnostics)
        if applied.tool_failure:
            return _respond(
                "TOOL_FAILURE", scenario, config, trace, request_id, maneuvers, failures, started_at, started_runtime
            )
        if applied.failure is not None:
            failures.append(applied.failure)
            state_failures.append(applied.failure)
            continue
        sandbox, pf_result = applied.sandbox, applied.result
        assert sandbox is not None and pf_result is not None
        maneuvers.append(maneuver)
        if _is_successful_terminal(pf_result):
            discard_sandbox(sandbox)
            return _respond(
                "SUCCESS",
                scenario,
                config,
                trace,
                request_id,
                maneuvers,
                failures,
                started_at,
                started_runtime,
                pf_result,
            )
        # Same registry hygiene as the agent loop: keep the net object, drop the handle.
        working = resolve_net(sandbox)
        discard_sandbox(sandbox)
        diagnostics = pf_result.diagnostics
        if pf_result.converged:
            feedback = _failure(
                iteration,
                "SOLVED_INFEASIBLE",
                None,
                _non_voltage_failure_detail(pf_result),
                maneuver,
            )
            failures.append(feedback)
            state_failures[:] = [feedback]
        else:
            failures.append(
                _failure(
                    iteration,
                    "STILL_DIVERGED",
                    pf_result.diagnostics,
                    pf_result.error_message,
                    maneuver,
                )
            )
            state_failures.clear()
    return _respond(
        "BUDGET_EXHAUSTED", scenario, config, trace, request_id, maneuvers, failures, started_at, started_runtime
    )


def _call_agent(
    agent_step: AgentStep,
    card: str,
    grid: SandboxNet | None,
    diagnostics: NRDiagnostics | None,
    history: Sequence[AcceptedManeuver],
    failures: Sequence[FailureFeedback],
    config: OrchestratorConfig,
    trace: TraceRecorder,
    iteration: int,
) -> AgentCallOutcome:
    started = time.monotonic()
    try:
        result = agent_step(
            card=card,
            grid=grid,
            diagnostics=diagnostics,
            history=history,
            failures=failures,
            config=config,
        )
    except LLMFailureError as exc:
        trace.add_llm_responses(llm_responses_of(exc), iteration=iteration)
        trace.record("diagnosis", "agent_step", {"iteration": iteration, "outcome": "LLM_FAILURE"}, started)
        return AgentCallOutcome(step_result=None, status="LLM_FAILURE")
    except ValidationError as exc:
        trace.add_llm_responses(llm_responses_of(exc), iteration=iteration)
        trace.record("diagnosis", "agent_step", {"iteration": iteration, "outcome": "MALFORMED_OUTPUT"}, started)
        return AgentCallOutcome(step_result=None, status="MALFORMED_OUTPUT", detail=str(exc))
    if not isinstance(result, AgentStepResult) or not isinstance(result.maneuver, Maneuver):
        trace.record("diagnosis", "agent_step", {"iteration": iteration, "outcome": "MALFORMED_OUTPUT"}, started)
        return AgentCallOutcome(step_result=None, status="MALFORMED_OUTPUT", detail="agent returned no maneuver")
    trace.add_llm_responses(result.llm_responses, iteration=iteration)
    trace.record("diagnosis", "agent_step", {"iteration": iteration, "llm_calls": len(result.llm_responses)}, started)
    return AgentCallOutcome(step_result=result, status="OK")


def _agent_grid(config: OrchestratorConfig, working: Any) -> SandboxNet | None:
    if config.CONFIGURATION not in {2, 3, 4, 5}:
        return None
    return create_sandbox(working)


def _apply_and_solve(
    trace: TraceRecorder,
    request_id: UUID,
    working: Any,
    maneuver: Maneuver,
    iteration: int,
    diagnostics: NRDiagnostics | None,
) -> ApplyOutcome:
    sandbox: Any | None = None
    started = time.monotonic()
    try:
        trace.n_tool_calls += 1
        sandbox = create_sandbox(working, scenario_request_id=request_id)
        trace.n_tool_calls += 1
        started = time.monotonic()
        saturated_gens = frozenset(diagnostics.gens_at_q_limit) if diagnostics is not None else frozenset()
        apply_maneuver(sandbox, maneuver, saturated_gens=saturated_gens)
        trace.record("maneuver", "apply", {"iteration": iteration}, started)
    except InvalidActionError as exc:
        assert sandbox is not None
        return _handle_invalid(trace, sandbox, maneuver, iteration, diagnostics, exc, started)
    except ToolFailureError:
        _discard_after_tool_failure(trace, sandbox)
        return ApplyOutcome(sandbox=None, result=None, tool_failure=True)
    try:
        result = _run_pf(sandbox, trace, "solve", "run_ac_pf", {"iteration": iteration})
        trace.n_tool_calls += 1
        promote_sandbox(sandbox)
    except ToolFailureError:
        return ApplyOutcome(sandbox=None, result=None, tool_failure=True)
    return ApplyOutcome(sandbox=sandbox, result=result)


def _handle_invalid(
    trace: TraceRecorder,
    sandbox: Any,
    maneuver: Maneuver,
    iteration: int,
    diagnostics: NRDiagnostics | None,
    exc: InvalidActionError,
    started: float,
) -> ApplyOutcome:
    trace.record("maneuver", "invalid_action", {"iteration": iteration, "detail": str(exc)}, started)
    trace.n_tool_calls += 1
    discard_sandbox(sandbox)
    feedback = _failure(iteration, "INVALID_ACTION", diagnostics, str(exc), maneuver)
    return ApplyOutcome(sandbox=None, result=None, failure=feedback)


def _run_pf(
    net: Any,
    trace: TraceRecorder,
    phase: Phase,
    event_name: str,
    payload: dict[str, Any] | None = None,
    *,
    record_diagnostics: bool = False,
) -> PowerFlowResult:
    trace.n_power_flows += 1
    started = time.monotonic()
    result = run_ac_pf(net)
    event_payload = dict(payload or {})
    # Only the baseline event carries the diagnostics dump; per-iteration solve events stay lean.
    if record_diagnostics and result.diagnostics is not None:
        event_payload["diagnostics"] = result.diagnostics.model_dump(mode="json")
    trace.record(phase, event_name, event_payload, started)
    return result


def _respond(
    status: Status,
    scenario: Scenario,
    config: OrchestratorConfig,
    trace: TraceRecorder,
    request_id: UUID,
    maneuvers: list[Maneuver],
    failures: list[FailureFeedback],
    started_at: datetime,
    started_runtime: float,
    final_result: PowerFlowResult | None = None,
) -> ResolutionResponse:
    started = time.monotonic()
    trace.record("response", "respond", {"status": status}, started)
    completed_at = datetime.now(timezone.utc)
    target_versioned = scenario.dataset_version is not None
    return ResolutionResponse(
        request_id=request_id,
        scenario_id=scenario.scenario_id,
        configuration=config.CONFIGURATION,
        llm_assignment=config.LLM_ASSIGNMENT.model_dump(exclude_none=True),
        repetition_index=config.repetition_index,
        dataset_version=scenario.dataset_version,
        solver_version=SOLVER_PROBE_POLICY_VERSION if target_versioned else None,
        action_policy_version=ACTION_POLICY_VERSION if target_versioned else None,
        ranking_policy_version=RANKING_POLICY_VERSION if target_versioned else None,
        result_schema_version=RESULT_SCHEMA_VERSION if target_versioned else None,
        status=status,
        maneuvers=maneuvers,
        n_maneuvers=len(maneuvers),
        converged=status == "SUCCESS",
        quality=final_result.quality if status == "SUCCESS" and final_result is not None else None,
        final_warnings=final_result.warnings if status == "SUCCESS" and final_result is not None else [],
        diagnosis_rationale=None,
        citations=[],
        failure_feedback=failures,
        trace=trace.build(),
        total_runtime_seconds=max(0.0, time.monotonic() - started_runtime),
        started_at=started_at,
        completed_at=completed_at,
    )


def _failure(
    iteration: int,
    kind: Literal[
        "STILL_DIVERGED",
        "SOLVED_INFEASIBLE",
        "INVALID_ACTION",
        "MALFORMED_OUTPUT",
        "LLM_FAILURE",
    ],
    diagnostics: NRDiagnostics | None,
    detail: str | None,
    maneuver: Maneuver | None,
) -> FailureFeedback:
    return FailureFeedback(iteration=iteration, kind=kind, diagnostics=diagnostics, detail=detail, maneuver=maneuver)


def _is_successful_terminal(result: PowerFlowResult) -> bool:
    """Apply the same non-voltage terminal contract used by offline witnesses."""
    return result.converged and satisfies_non_voltage_constraints(
        result.feasibility
    )


def _non_voltage_failure_detail(result: PowerFlowResult) -> str:
    feasibility = result.feasibility
    if feasibility is None:
        return "power flow converged without the feasibility evidence required for benchmark success"
    reasons = [
        reason.detail
        for reason in feasibility.failure_reasons
        if reason.code != "HARD_VOLTAGE_ENVELOPE"
    ]
    if reasons:
        return "; ".join(reasons)
    return "power flow converged but violates a non-voltage benchmark constraint"


def _timed_out(started_runtime: float, config: OrchestratorConfig) -> bool:
    return time.monotonic() - started_runtime > config.MAX_RUNTIME_SECONDS


def _elapsed_ms(started: float) -> float:
    return max(0.0, (time.monotonic() - started) * 1000.0)


def _discard_after_tool_failure(trace: TraceRecorder, sandbox: Any | None) -> None:
    if sandbox is None:
        return
    trace.n_tool_calls += 1
    discard_sandbox(sandbox)
