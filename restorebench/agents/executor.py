# ABOUTME: Implements the Executor role and orchestrator-agent prompt helpers for multi-agent cases.
# ABOUTME: Grounds proposed maneuvers with deterministic diagnostic tools before returning reports.
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import ValidationError

from restorebench.agents.prompt_fragments import ACTION_VOCABULARY_AND_BOUNDS, COMPOSITION_AND_PROGRESS
from restorebench.agents.tool_loop import (
    MAX_DIAGNOSTIC_TOOL_CALLS,
    TerminalTool,
    default_diagnostic_tools,
    retry_messages,
    run_tool_loop,
)
from restorebench.llm.providers import ChatMessage, LLMResponse, ToolUse, llm_call
from restorebench.schemas.actions import Action, Maneuver
from restorebench.schemas.config import OrchestratorConfig
from restorebench.schemas.feedback import FailedManeuverAttempt, FailureFeedback, SandboxNet
from restorebench.schemas.errors import InvalidActionError, LLMFailureError, attach_llm_responses, llm_responses_of
from restorebench.schemas.multi_agent import AnalystAssessment, ExecutorReport
from restorebench.schemas.power_flow import NRDiagnostics
from restorebench.schemas.topology import ApplicabilityResult
from restorebench.tools.power_flow import run_ac_pf
from restorebench.tools.sandbox import apply_maneuver, create_sandbox, discard_sandbox, resolve_net
from restorebench.tools.topology import get_action_applicability
from restorebench.agents.structured_output import coerce_nested_json


PROPOSE_MANEUVER_TOOL_NAME = "propose_maneuver"


@dataclass(frozen=True)
class ExecutorResult:
    report: ExecutorReport
    llm_responses: tuple[LLMResponse, ...]
    failed_attempts: tuple[FailedManeuverAttempt, ...] = ()


class ExecutorRole(Protocol):
    def __call__(
        self,
        *,
        card: str,
        grid: SandboxNet,
        assessment: AnalystAssessment,
        config: OrchestratorConfig,
        saturated_gens: frozenset[int] = frozenset(),
        failed_actions: Sequence[Action] = (),
    ) -> ExecutorResult: ...


def make_executor(model_id: str) -> ExecutorRole:
    def executor_step(
        *,
        card: str,
        grid: SandboxNet,
        assessment: AnalystAssessment,
        config: OrchestratorConfig,
        saturated_gens: frozenset[int] = frozenset(),
        failed_actions: Sequence[Action] = (),
    ) -> ExecutorResult:
        messages = _build_messages(card, assessment, config)
        responses: list[LLMResponse] = []
        failed_attempts: list[FailedManeuverAttempt] = []
        # On failure carry the responses already produced (this call's plus any the raising tool loop
        # attached) so the orchestrator can still bill the tokens they cost.
        try:
            for attempt in range(2):
                loop_result = run_tool_loop(
                    model_id=model_id,
                    messages=messages,
                    grid=grid,
                    diagnostic_tools=default_diagnostic_tools(
                        saturated_gens=saturated_gens,
                        failed_actions=(
                            *failed_actions,
                            *(attempt.maneuver.action for attempt in failed_attempts),
                        ),
                        failed_attempts=failed_attempts,
                    ),
                    terminal_tool=_maneuver_terminal_tool(),
                    max_diagnostic_tool_calls=MAX_DIAGNOSTIC_TOOL_CALLS,
                    role="executor",
                    llm_call_fn=llm_call,
                )
                responses.extend(loop_result.responses)
                try:
                    maneuver = Maneuver.model_validate(
                        coerce_nested_json(loop_result.tool_use.input, Maneuver)
                    )
                except ValidationError as exc:
                    if attempt == 1:
                        raise
                    messages.extend(
                        _reprompt_messages(loop_result.responses[-1], exc, loop_result.tool_use)
                    )
                    continue
                report = build_executor_report(
                    grid,
                    maneuver,
                    saturated_gens=saturated_gens,
                    failed_actions=(*failed_actions, *(attempt.maneuver.action for attempt in failed_attempts)),
                )
                return ExecutorResult(
                    report=report,
                    llm_responses=tuple(responses),
                    failed_attempts=tuple(failed_attempts),
                )
            raise AssertionError("unreachable")
        except (LLMFailureError, ValidationError) as exc:
            attach_llm_responses(exc, (*responses, *llm_responses_of(exc)))
            raise

    return executor_step


def build_executor_report(
    grid: SandboxNet,
    maneuver: Maneuver,
    *,
    saturated_gens: frozenset[int] = frozenset(),
    failed_actions: Sequence[Action] = (),
) -> ExecutorReport:
    if any(action == maneuver.action for action in failed_actions):
        applicability = ApplicabilityResult(
            action=maneuver.action,
            applicable=False,
            reason="the same action already failed on the current grid state",
        )
        return ExecutorReport(maneuver=maneuver, applicability=applicability, pf_result=run_ac_pf(grid))
    applicability = get_action_applicability(grid, maneuver.action, saturated_gens=saturated_gens)
    if not applicability.applicable:
        # An inapplicable proposal still yields a report: the orchestrator-agent sees the
        # verdict, and if it commits anyway the runner burns an INVALID_ACTION budget slot.
        # Applying would just raise InvalidActionError, so the preview solves the CURRENT grid.
        return ExecutorReport(maneuver=maneuver, applicability=applicability, pf_result=run_ac_pf(grid))

    sandbox = create_sandbox(resolve_net(grid))
    try:
        apply_maneuver(sandbox, maneuver, saturated_gens=saturated_gens)
        pf_result = run_ac_pf(sandbox)
    except InvalidActionError as exc:
        # Belt and braces: apply-time bounds can be stricter than the pre-screen. This is
        # strategy feedback, never a bug — the orchestrator only maps ValidationError and
        # LLMFailureError, so letting this escape would crash resolve() with no checkpoint.
        applicability = ApplicabilityResult(action=maneuver.action, applicable=False, reason=str(exc))
        pf_result = run_ac_pf(grid)
    finally:
        discard_sandbox(sandbox)
    return ExecutorReport(maneuver=maneuver, applicability=applicability, pf_result=pf_result)


def _maneuver_terminal_tool() -> TerminalTool:
    return TerminalTool(
        name=PROPOSE_MANEUVER_TOOL_NAME,
        description="Return the concrete Maneuver the deterministic runner should try for this iteration.",
        input_schema=Maneuver.model_json_schema(),
        output_model=Maneuver,
    )


def executor_tool_config() -> dict[str, Any]:
    from restorebench.agents.tool_loop import build_tool_config

    return build_tool_config(default_diagnostic_tools(), _maneuver_terminal_tool())


def _reprompt_messages(
    response: LLMResponse,
    exc: ValidationError,
    tool_use: ToolUse,
) -> list[ChatMessage]:
    detail = (
        "Your previous response did not validate as a Maneuver. "
        "Call propose_maneuver with valid JSON matching the schema. "
        f"Validation error:\n{exc}"
    )
    return retry_messages(response, detail=detail, tool_use=tool_use)


def _build_messages(card: str, assessment: AnalystAssessment, config: OrchestratorConfig) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=_system_prompt(config)),
        ChatMessage(role="user", content=_user_prompt(card, assessment)),
    ]


def _system_prompt(config: OrchestratorConfig) -> str:
    return f"""You are the Executor in a multi-agent grid-resolution team.
Turn the AnalystAssessment into one concrete bounds-respecting Maneuver.
Use diagnostic tools to verify applicability before finalizing.
The deterministic runner will apply the final maneuver and decide convergence.

{ACTION_VOCABULARY_AND_BOUNDS}

{COMPOSITION_AND_PROGRESS}

rank_candidate_maneuvers surveys the grid for you: it evaluates a spread of legal maneuvers and returns them ordered by how much overstress each removes. Call it first.
run_ac_pf is your verification tool: pass a candidate maneuver and read 'converged' to see if it fixes the divergence. Test candidates with it before finalizing; calling it with no maneuver only re-previews the known-diverging base case.
Use at most {MAX_DIAGNOSTIC_TOOL_CALLS} diagnostic tool calls. The final answer must be a propose_maneuver tool call.
MANEUVER_BUDGET={config.MANEUVER_BUDGET}; this call finalizes one maneuver for the current iteration."""


def _user_prompt(card: str, assessment: AnalystAssessment) -> str:
    return "\n\n".join(
        [
            "# Scenario Card",
            card,
            "# AnalystAssessment",
            json.dumps(assessment.model_dump(mode="json"), indent=2, sort_keys=True),
        ]
    )


def orchestrator_agent_messages(
    *,
    assessment: AnalystAssessment,
    report: ExecutorReport,
    diagnostics: NRDiagnostics | None,
    failures: Sequence[FailureFeedback],
    config: OrchestratorConfig,
) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=_orchestrator_system_prompt(config)),
        ChatMessage(role="user", content=_orchestrator_user_prompt(assessment, report, diagnostics, failures)),
    ]


def _orchestrator_system_prompt(config: OrchestratorConfig) -> str:
    return f"""You are the orchestrator_agent in a multi-agent grid-resolution team.
Review the structured AnalystAssessment and ExecutorReport for this iteration.
Return COMMIT when the executor maneuver should be handed to the deterministic runner, or REVISE with concise guidance for one revised Analyst call.
Do not use hidden reasoning or free text as an input to other roles; coordinate only through the structured reports.

{COMPOSITION_AND_PROGRESS}

MANEUVER_BUDGET={config.MANEUVER_BUDGET}; this decision covers one maneuver for the current iteration."""


def _orchestrator_user_prompt(
    assessment: AnalystAssessment,
    report: ExecutorReport,
    diagnostics: NRDiagnostics | None,
    failures: Sequence[FailureFeedback],
) -> str:
    return "\n\n".join(
        [
            "# Latest NR diagnostics",
            "null"
            if diagnostics is None
            else json.dumps(diagnostics.model_dump(mode="json"), indent=2, sort_keys=True),
            "# Failed attempts on current grid state (do not repeat their actions)",
            _json_lines([failure.model_dump(mode="json") for failure in failures]),
            "# AnalystAssessment",
            json.dumps(assessment.model_dump(mode="json"), indent=2, sort_keys=True),
            "# ExecutorReport",
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True),
        ]
    )


def _json_lines(payload: list[dict[str, Any]]) -> str:
    if not payload:
        return "[]"
    return json.dumps(payload, indent=2, sort_keys=True)
