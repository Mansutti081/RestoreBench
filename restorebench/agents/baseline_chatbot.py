# ABOUTME: Implements the Case-1 one-shot chatbot proposer.
# ABOUTME: Produces a full ManeuverSequence without diagnostic tools or solver feedback.
from __future__ import annotations

from typing import Any, NoReturn

from pydantic import ValidationError

from restorebench.agents.prompt_fragments import ACTION_VOCABULARY_AND_BOUNDS, CAUSE_ACTION_PRIOR, CAUSE_TAXONOMY
from restorebench.environment.orchestrator import ProposeSequence, ProposeSequenceResult
from restorebench.agents.tool_loop import retry_messages
from restorebench.llm.providers import ChatMessage, LLMResponse, ToolUse, llm_call
from restorebench.schemas.actions import ManeuverSequence
from restorebench.schemas.config import OrchestratorConfig
from restorebench.schemas.errors import LLMFailureError, attach_llm_responses
from restorebench.agents.structured_output import coerce_nested_json


PROPOSE_SEQUENCE_TOOL_NAME = "propose_maneuver_sequence"


def make_baseline_chatbot(model_id: str) -> ProposeSequence:
    """Return the Case-1 chatbot: one sequence proposal, with one malformed-output retry."""

    def propose_sequence(*, card: str, config: OrchestratorConfig) -> ProposeSequenceResult:
        messages = _build_messages(card, config)
        responses: list[LLMResponse] = []

        # A truncation/throttle or a final validation failure raises mid-loop; carry the responses
        # already produced so the orchestrator can still bill the tokens they cost.
        try:
            for attempt in range(2):
                response = _call_model(model_id, messages)
                response.raw["role"] = "chatbot"
                responses.append(response)
                try:
                    sequence = _sequence_from_response(response)
                except ValidationError as exc:
                    if attempt == 1:
                        raise
                    messages.extend(_reprompt_messages(response, exc))
                    continue
                return ProposeSequenceResult(maneuvers=sequence, llm_responses=tuple(responses))
            _raise_missing_sequence()
        except (LLMFailureError, ValidationError) as exc:
            attach_llm_responses(exc, responses)
            raise

    return propose_sequence


def _call_model(model_id: str, messages: list[ChatMessage]) -> LLMResponse:
    return llm_call(
        model_id,
        messages,
        temperature=1.0,
        thinking=True,
        tools=baseline_chatbot_tool_config(),
    )


def baseline_chatbot_tool_config() -> dict[str, Any]:
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": PROPOSE_SEQUENCE_TOOL_NAME,
                    "description": "Return an ordered ManeuverSequence to try for restoring AC power-flow convergence.",
                    "inputSchema": {"json": ManeuverSequence.model_json_schema()},
                }
            }
        ]
    }


def _sequence_from_response(response: LLMResponse) -> ManeuverSequence:
    proposal = _proposal_tool(response.tool_uses)
    payload = proposal.input if proposal is not None else {}
    return ManeuverSequence.model_validate(coerce_nested_json(payload, ManeuverSequence))


def _proposal_tool(tool_uses: tuple[ToolUse, ...]) -> ToolUse | None:
    return next((tool_use for tool_use in tool_uses if tool_use.name == PROPOSE_SEQUENCE_TOOL_NAME), None)


def _reprompt_messages(response: LLMResponse, exc: ValidationError) -> list[ChatMessage]:
    detail = (
        "Your previous response did not validate as a ManeuverSequence. "
        "Call propose_maneuver_sequence with valid JSON matching the schema. "
        f"Validation error:\n{exc}"
    )
    return retry_messages(response, detail=detail, tool_use=_proposal_tool(response.tool_uses))


def _build_messages(card: str, config: OrchestratorConfig) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=_system_prompt(config)),
        ChatMessage(role="user", content=_user_prompt(card)),
    ]


def _system_prompt(config: OrchestratorConfig) -> str:
    return f"""You are the Case-1 baseline grid-resolution chatbot. Propose an ordered ManeuverSequence up front.

First reconstruct the relevant grid structure from the Scenario Card before choosing the maneuver sequence.
Fill reconstruction_summary with the concise grid reconstruction and strategy you used.
Your maneuvers are applied in order by a deterministic operator, stopping at first convergence, with no feedback between maneuvers.
They accumulate: each maneuver is applied on top of the ones before it and stays applied, so the sequence is one cumulative plan rather than a list of independent alternatives. Many cases admit no single action that converges and are solvable only by several stacked maneuvers.
Do not propose topology switching, load changes, adding/removing components, or actions outside the allowed schema.

{ACTION_VOCABULARY_AND_BOUNDS}

{CAUSE_TAXONOMY}

{CAUSE_ACTION_PRIOR}

Use only the declared atomic Q-V actions.
MANEUVER_BUDGET={config.MANEUVER_BUDGET}; propose at most this many maneuvers, ordered as the cumulative plan should be applied."""


def _user_prompt(card: str) -> str:
    return "\n\n".join(
        [
            "# Scenario Card",
            card,
            "# Non-convergence statement",
            "The AC power flow does not converge for this scenario.",
        ]
    )


def _raise_missing_sequence() -> NoReturn:
    ManeuverSequence.model_validate({})
    raise AssertionError("unreachable")
