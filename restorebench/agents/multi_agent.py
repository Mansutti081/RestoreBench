# ABOUTME: Composes Analyst, Executor, and orchestrator-agent into one AgentStep.
# ABOUTME: Runs the analyst/executor/orchestrator roles and merges their outputs.
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, NoReturn

from pydantic import BaseModel, ConfigDict, ValidationError

from restorebench.agents.analyst import make_analyst
from restorebench.agents.executor import make_executor, orchestrator_agent_messages
from restorebench.environment.orchestrator import AgentStep, AgentStepResult
from restorebench.agents.tool_loop import retry_messages
from restorebench.llm.providers import ChatMessage, LLMResponse, ToolUse, llm_call
from restorebench.schemas.errors import LLMFailureError, attach_llm_responses, llm_responses_of
from restorebench.schemas.actions import Action
from restorebench.schemas.config import OrchestratorConfig
from restorebench.schemas.feedback import AcceptedManeuver, FailedManeuverAttempt, FailureFeedback, SandboxNet
from restorebench.schemas.multi_agent import AnalystAssessment, ExecutorReport
from restorebench.schemas.power_flow import NRDiagnostics
from restorebench.agents.structured_output import coerce_nested_json


MAX_REVISION_ROUNDS = 1
ORCHESTRATOR_DECISION_TOOL_NAME = "submit_orchestrator_decision"


class OrchestratorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["COMMIT", "REVISE"]
    guidance: str | None = None


@dataclass(frozen=True)
class RoleOutputs:
    assessment: AnalystAssessment
    report: ExecutorReport
    llm_responses: tuple[LLMResponse, ...]
    failed_attempts: tuple[FailedManeuverAttempt, ...] = ()


@dataclass(frozen=True)
class OrchestratorCall:
    decision: OrchestratorDecision
    llm_responses: tuple[LLMResponse, ...]


def make_multi_agent() -> AgentStep:
    def multi_agent_step(
        *,
        card: str,
        grid: SandboxNet | None,
        diagnostics: NRDiagnostics | None,
        history: Sequence[AcceptedManeuver],
        failures: Sequence[FailureFeedback],
        config: OrchestratorConfig,
    ) -> AgentStepResult:
        if grid is None:
            raise RuntimeError("multi_agent requires a diagnostic grid sandbox")
        _validate_configuration(config)
        assignment = config.LLM_ASSIGNMENT
        assert (
            assignment.analyst is not None and assignment.executor is not None and assignment.orchestrator is not None
        )

        responses: list[LLMResponse] = []
        failed_attempts: list[FailedManeuverAttempt] = []
        # Roles and revision rounds each bill tokens; if a later one fails, merge everything already
        # collected with the responses the failing role attached so none are lost with the exception.
        try:
            outputs = _run_roles(
                card,
                grid,
                diagnostics,
                history,
                failures,
                config,
                assignment.analyst,
                assignment.executor,
                revision_guidance=None,
                failed_actions=_failed_actions(failures, failed_attempts),
            )
            responses.extend(outputs.llm_responses)
            failed_attempts.extend(outputs.failed_attempts)
            decision = _call_orchestrator_agent(
                assignment.orchestrator, outputs.assessment, outputs.report, diagnostics, failures, config
            )
            responses.extend(decision.llm_responses)

            for _round in range(MAX_REVISION_ROUNDS):
                if decision.decision.decision != "REVISE":
                    break
                rejected_attempt = _failed_report_attempt(outputs.report)
                if rejected_attempt is not None and not any(
                    attempt.maneuver.action == rejected_attempt.maneuver.action for attempt in failed_attempts
                ):
                    failed_attempts.append(rejected_attempt)
                outputs = _run_roles(
                    card,
                    grid,
                    diagnostics,
                    history,
                    failures,
                    config,
                    assignment.analyst,
                    assignment.executor,
                    revision_guidance=decision.decision.guidance or "Revise the proposed maneuver.",
                    failed_actions=_failed_actions(failures, failed_attempts),
                )
                responses.extend(outputs.llm_responses)
                failed_attempts.extend(outputs.failed_attempts)
                decision = _call_orchestrator_agent(
                    assignment.orchestrator, outputs.assessment, outputs.report, diagnostics, failures, config
                )
                responses.extend(decision.llm_responses)
        except (LLMFailureError, ValidationError) as exc:
            attach_llm_responses(exc, (*responses, *llm_responses_of(exc)))
            raise

        maneuver = outputs.report.maneuver
        return AgentStepResult(
            maneuver=maneuver,
            llm_responses=tuple(responses),
            failed_attempts=tuple(failed_attempts),
        )

    return multi_agent_step


def _validate_configuration(config: OrchestratorConfig) -> None:
    if config.CONFIGURATION != 3:
        raise RuntimeError("multi_agent requires CONFIGURATION 3")
    assignment = config.LLM_ASSIGNMENT
    if assignment.analyst is None or assignment.executor is None or assignment.orchestrator is None:
        raise RuntimeError("multi_agent requires analyst, executor, and orchestrator model assignments")


def _run_roles(
    card: str,
    grid: SandboxNet,
    diagnostics: NRDiagnostics | None,
    history: Sequence[AcceptedManeuver],
    failures: Sequence[FailureFeedback],
    config: OrchestratorConfig,
    analyst_model: str,
    executor_model: str,
    revision_guidance: str | None,
    failed_actions: Sequence[Action],
) -> RoleOutputs:
    analyst_result = make_analyst(analyst_model)(
        card=card,
        diagnostics=diagnostics,
        history=history,
        failures=failures,
        config=config,
        revision_guidance=revision_guidance,
    )
    # The analyst already succeeded and billed tokens; if the executor now fails, prepend the
    # analyst's responses so they are not lost with the raising exception.
    saturated_gens = frozenset(diagnostics.gens_at_q_limit) if diagnostics is not None else frozenset()
    try:
        executor_result = make_executor(executor_model)(
            card=card,
            grid=grid,
            assessment=analyst_result.assessment,
            config=config,
            saturated_gens=saturated_gens,
            failed_actions=failed_actions,
        )
    except (LLMFailureError, ValidationError) as exc:
        attach_llm_responses(exc, (*analyst_result.llm_responses, *llm_responses_of(exc)))
        raise
    return RoleOutputs(
        assessment=analyst_result.assessment,
        report=executor_result.report,
        llm_responses=(*analyst_result.llm_responses, *executor_result.llm_responses),
        failed_attempts=executor_result.failed_attempts,
    )


def _failed_actions(
    failures: Sequence[FailureFeedback],
    attempts: Sequence[FailedManeuverAttempt],
) -> tuple[Action, ...]:
    return (
        *(failure.maneuver.action for failure in failures if failure.maneuver is not None),
        *(attempt.maneuver.action for attempt in attempts),
    )


def _failed_report_attempt(report: ExecutorReport) -> FailedManeuverAttempt | None:
    if not report.applicability.applicable:
        return FailedManeuverAttempt(
            kind="PREVIEW_INVALID",
            maneuver=report.maneuver,
            diagnostics=None,
            detail=report.applicability.reason,
        )
    if not report.pf_result.converged:
        return FailedManeuverAttempt(
            kind="PREVIEW_DIVERGED",
            maneuver=report.maneuver,
            diagnostics=report.pf_result.diagnostics,
            detail=report.pf_result.error_message,
        )
    return None


def _call_orchestrator_agent(
    model_id: str,
    assessment: AnalystAssessment,
    report: ExecutorReport,
    diagnostics: NRDiagnostics | None,
    failures: Sequence[FailureFeedback],
    config: OrchestratorConfig,
) -> OrchestratorCall:
    messages = orchestrator_agent_messages(
        assessment=assessment,
        report=report,
        diagnostics=diagnostics,
        failures=failures,
        config=config,
    )
    responses: list[LLMResponse] = []
    try:
        for attempt in range(2):
            response = llm_call(model_id, messages, temperature=1.0, thinking=True, tools=_orchestrator_tool_config())
            response.raw["role"] = "orchestrator_agent"
            responses.append(response)
            try:
                decision = _decision_from_response(response)
            except ValidationError as exc:
                if attempt == 1:
                    raise
                messages.extend(_reprompt_messages(response, exc))
                continue
            return OrchestratorCall(decision=decision, llm_responses=tuple(responses))
        _raise_missing_decision()
    except (LLMFailureError, ValidationError) as exc:
        attach_llm_responses(exc, responses)
        raise


def _orchestrator_tool_config() -> dict[str, Any]:
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": ORCHESTRATOR_DECISION_TOOL_NAME,
                    "description": "Commit the ExecutorReport maneuver or request one Analyst revision.",
                    "inputSchema": {"json": OrchestratorDecision.model_json_schema()},
                }
            }
        ]
    }


def _decision_from_response(response: LLMResponse) -> OrchestratorDecision:
    proposal = _proposal_tool(response.tool_uses)
    payload = proposal.input if proposal is not None else {}
    return OrchestratorDecision.model_validate(coerce_nested_json(payload, OrchestratorDecision))


def _proposal_tool(tool_uses: Sequence[ToolUse]) -> ToolUse | None:
    return next((tool_use for tool_use in tool_uses if tool_use.name == ORCHESTRATOR_DECISION_TOOL_NAME), None)


def _reprompt_messages(response: LLMResponse, exc: ValidationError) -> list[ChatMessage]:
    detail = (
        "Your previous response did not validate as an OrchestratorDecision. "
        "Call submit_orchestrator_decision with valid JSON matching the schema. "
        f"Validation error:\n{exc}"
    )
    return retry_messages(response, detail=detail, tool_use=_proposal_tool(response.tool_uses))


def _raise_missing_decision() -> NoReturn:
    OrchestratorDecision.model_validate({})
    raise AssertionError("unreachable")
