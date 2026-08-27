# ABOUTME: Implements the Analyst role for multi-agent configurations 3 and 5.
# ABOUTME: Produces structured diagnosis and proposed maneuvers without touching diagnostic tools.
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NoReturn, Protocol

from pydantic import ValidationError

from restorebench.agents.prompt_fragments import (
    ACTION_VOCABULARY_AND_BOUNDS,
    CAUSE_ACTION_PRIOR,
    CAUSE_TAXONOMY,
    COMPOSITION_AND_PROGRESS,
)
from restorebench.agents.history_render import render_accepted_history
from restorebench.agents.tool_loop import retry_messages
from restorebench.llm.providers import ChatMessage, LLMResponse, ToolUse, llm_call
from restorebench.schemas.config import OrchestratorConfig
from restorebench.schemas.errors import LLMFailureError, attach_llm_responses
from restorebench.schemas.feedback import AcceptedManeuver, FailureFeedback
from restorebench.schemas.multi_agent import AnalystAssessment
from restorebench.schemas.power_flow import NRDiagnostics
from restorebench.agents.structured_output import coerce_nested_json


ANALYST_TOOL_NAME = "submit_analyst_assessment"


@dataclass(frozen=True)
class AnalystResult:
    assessment: AnalystAssessment
    llm_responses: tuple[LLMResponse, ...]


class AnalystRole(Protocol):
    def __call__(
        self,
        *,
        card: str,
        diagnostics: NRDiagnostics | None,
        history: Sequence[AcceptedManeuver],
        failures: Sequence[FailureFeedback],
        config: OrchestratorConfig,
        revision_guidance: str | None = None,
    ) -> AnalystResult: ...


def make_analyst(model_id: str) -> AnalystRole:
    def analyst_step(
        *,
        card: str,
        diagnostics: NRDiagnostics | None,
        history: Sequence[AcceptedManeuver],
        failures: Sequence[FailureFeedback],
        config: OrchestratorConfig,
        revision_guidance: str | None = None,
    ) -> AnalystResult:
        messages = _build_messages(card, diagnostics, history, failures, config, revision_guidance)
        responses: list[LLMResponse] = []
        # A truncation/throttle or a final validation failure raises mid-loop; carry the responses
        # already produced so the orchestrator can still bill the tokens they cost.
        try:
            for attempt in range(2):
                response = _call_model(model_id, messages)
                response.raw["role"] = "analyst"
                responses.append(response)
                try:
                    assessment = _assessment_from_response(response)
                except ValidationError as exc:
                    if attempt == 1:
                        raise
                    messages.extend(_reprompt_messages(response, exc))
                    continue
                return AnalystResult(assessment=assessment, llm_responses=tuple(responses))
            _raise_missing_assessment()
        except (LLMFailureError, ValidationError) as exc:
            attach_llm_responses(exc, responses)
            raise

    return analyst_step


def _call_model(model_id: str, messages: list[ChatMessage]) -> LLMResponse:
    return llm_call(
        model_id,
        messages,
        temperature=1.0,
        thinking=True,
        tools=analyst_tool_config(),
    )


def analyst_tool_config() -> dict[str, Any]:
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": ANALYST_TOOL_NAME,
                    "description": "Return the diagnosis and proposed maneuver as an AnalystAssessment.",
                    "inputSchema": {"json": AnalystAssessment.model_json_schema()},
                }
            }
        ]
    }


def _assessment_from_response(response: LLMResponse) -> AnalystAssessment:
    proposal = _proposal_tool(response.tool_uses)
    payload = proposal.input if proposal is not None else {}
    assessment = AnalystAssessment.model_validate(coerce_nested_json(payload, AnalystAssessment))
    return assessment


def _proposal_tool(tool_uses: Sequence[ToolUse]) -> ToolUse | None:
    return next((tool_use for tool_use in tool_uses if tool_use.name == ANALYST_TOOL_NAME), None)


def _reprompt_messages(response: LLMResponse, exc: ValidationError) -> list[ChatMessage]:
    detail = (
        "Your previous response did not validate as an AnalystAssessment. "
        "Call submit_analyst_assessment with valid JSON matching the schema. "
        f"Validation error:\n{exc}"
    )
    return retry_messages(response, detail=detail, tool_use=_proposal_tool(response.tool_uses))


def _build_messages(
    card: str,
    diagnostics: NRDiagnostics | None,
    history: Sequence[AcceptedManeuver],
    failures: Sequence[FailureFeedback],
    config: OrchestratorConfig,
    revision_guidance: str | None,
) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=_system_prompt(config)),
        ChatMessage(
            role="user", content=_user_prompt(card, diagnostics, history, failures, revision_guidance)
        ),
    ]


def _system_prompt(config: OrchestratorConfig) -> str:
    prompt = f"""You are the Analyst in a multi-agent grid-resolution team.
Diagnose the non-convergence cause from the Scenario Card and latest NR diagnostics, then propose one maneuver for the Executor to verify.
First reconstruct the relevant grid structure from the Scenario Card before choosing the maneuver.

{ACTION_VOCABULARY_AND_BOUNDS}

{CAUSE_TAXONOMY}

{CAUSE_ACTION_PRIOR}

{COMPOSITION_AND_PROGRESS}

Use only the declared atomic Q-V actions.
MANEUVER_BUDGET={config.MANEUVER_BUDGET}; propose one maneuver for the current iteration."""
    return prompt


def _user_prompt(
    card: str,
    diagnostics: NRDiagnostics | None,
    history: Sequence[AcceptedManeuver],
    failures: Sequence[FailureFeedback],
    revision_guidance: str | None,
) -> str:
    sections = ["# Scenario Card", card, "# Latest NR diagnostics", _diagnostics_text(diagnostics)]
    if revision_guidance:
        sections.extend(["# Revision guidance from orchestrator-agent", revision_guidance])
    sections.extend(
        [
            "# Accepted maneuver history (each step is still applied; overstress is the distance from solvability)",
            render_accepted_history(history),
            "# Failed attempts on current grid state (do not repeat their actions)",
            _json_lines([failure.model_dump(mode="json") for failure in failures]),
        ]
    )
    return "\n\n".join(sections)


def _diagnostics_text(diagnostics: NRDiagnostics | None) -> str:
    if diagnostics is None:
        return "No diagnostics are available yet."
    return "\n".join(
        [
            f"iterations_attempted: {diagnostics.iterations_attempted}",
            f"worst_bus: {diagnostics.worst_bus}",
            f"lowest_vm_pu: {diagnostics.lowest_vm_pu}",
            f"lowest_vm_bus: {diagnostics.lowest_vm_bus}",
            f"gens_at_q_limit: {diagnostics.gens_at_q_limit}",
            f"max_mismatch_mw: {diagnostics.max_mismatch_mw}",
            f"max_mismatch_mvar: {diagnostics.max_mismatch_mvar}",
            f"overstress: {diagnostics.overstress}",
            f"error_message: {diagnostics.error_message}",
            f"diagnostics_source: {diagnostics.diagnostics_source}",
        ]
    )


def _json_lines(payload: list[dict[str, Any]]) -> str:
    if not payload:
        return "[]"
    return json.dumps(payload, indent=2, sort_keys=True)


def _raise_missing_assessment() -> NoReturn:
    AnalystAssessment.model_validate({})
    raise AssertionError("unreachable")
