# ABOUTME: Implements the Case-2 single-agent seam for one LLM-proposed maneuver per iteration.
# ABOUTME: Lets the agent probe deterministic diagnostic tools before the orchestrator applies the final maneuver.
from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, NoReturn

from pydantic import TypeAdapter, ValidationError

from restorebench.agents.history_render import render_accepted_history
from restorebench.agents.prompt_fragments import (
    ACTION_VOCABULARY_AND_BOUNDS,
    CAUSE_ACTION_PRIOR,
    CAUSE_TAXONOMY,
    COMPOSITION_AND_PROGRESS,
)
from restorebench.agents.tool_loop import MAX_DIAGNOSTIC_TOOL_CALLS, DiagnosticTool, TerminalTool, build_tool_config
from restorebench.agents.tool_loop import run_tool_loop
from restorebench.agents.tool_loop import default_diagnostic_tools
from restorebench.environment.orchestrator import AgentStep, AgentStepResult
from restorebench.llm.providers import ChatMessage, llm_call
from restorebench.schemas.actions import Action, Maneuver
from restorebench.schemas.config import OrchestratorConfig
from restorebench.schemas.errors import attach_llm_responses
from restorebench.schemas.feedback import AcceptedManeuver, FailedManeuverAttempt, FailureFeedback, SandboxNet
from restorebench.schemas.power_flow import NRDiagnostics
from restorebench.agents.structured_output import coerce_nested_json


PROPOSE_MANEUVER_TOOL_NAME = "propose_maneuver"
_ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)


def make_single_agent(model_id: str) -> AgentStep:
    """Return the Case-2 agent: one maneuver per iteration after up to three diagnostic tool calls."""

    def single_agent_step(
        *,
        card: str,
        grid: SandboxNet | None = None,
        diagnostics: NRDiagnostics | None,
        history: Sequence[AcceptedManeuver],
        failures: Sequence[FailureFeedback],
        config: OrchestratorConfig,
    ) -> AgentStepResult:
        if grid is None:
            raise RuntimeError("single_agent requires a diagnostic grid sandbox")

        messages = _build_messages(card, diagnostics, history, failures, config)
        failed_attempts: list[FailedManeuverAttempt] = []
        loop_result = run_tool_loop(
            model_id=model_id,
            messages=messages,
            grid=grid,
            diagnostic_tools=default_diagnostic_tools(
                saturated_gens=_saturated_gens(diagnostics),
                failed_actions=_failed_actions(failures),
                failed_attempts=failed_attempts,
            ),
            terminal_tool=_maneuver_terminal_tool(),
            max_diagnostic_tool_calls=MAX_DIAGNOSTIC_TOOL_CALLS,
            role="single_agent",
            llm_call_fn=llm_call,
        )
        try:
            maneuver = Maneuver.model_validate(
                coerce_nested_json(loop_result.tool_use.input, Maneuver)
            )
        except ValidationError as exc:
            attach_llm_responses(exc, loop_result.responses)
            raise
        return AgentStepResult(
            maneuver=maneuver,
            llm_responses=loop_result.responses,
            failed_attempts=tuple(failed_attempts),
        )

    return single_agent_step


def single_agent_tool_config() -> dict[str, Any]:
    return build_tool_config(_single_agent_diagnostic_tools(), _maneuver_terminal_tool())


maneuver_tool_config = single_agent_tool_config


def _maneuver_terminal_tool() -> TerminalTool:
    return TerminalTool(
        name=PROPOSE_MANEUVER_TOOL_NAME,
        description="Return exactly one Maneuver to try next for restoring AC power-flow convergence.",
        input_schema=Maneuver.model_json_schema(),
        output_model=Maneuver,
    )


def _single_agent_diagnostic_tools() -> tuple[DiagnosticTool, ...]:
    # The SingleAgent and the Executor probe the grid with the same three tools, so they
    # share one definition. A private copy here silently kept a Bedrock-invalid schema
    # alive after the shared one was fixed (the Action union needs an object wrapper).
    return default_diagnostic_tools()


def _saturated_gens(diagnostics: NRDiagnostics | None) -> frozenset[int]:
    # The applicability tool rejects raising the setpoint of these generators; they have switched
    # to PQ and cannot inject more reactive power. The diagnostics carry the current saturated set.
    if diagnostics is None:
        return frozenset()
    return frozenset(diagnostics.gens_at_q_limit)


def _failed_actions(failures: Sequence[FailureFeedback]) -> tuple[Action, ...]:
    return tuple(failure.maneuver.action for failure in failures if failure.maneuver is not None)


def _dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return {"value": value}


def _raise_missing_maneuver() -> NoReturn:
    Maneuver.model_validate({})
    raise AssertionError("unreachable")


def _build_messages(
    card: str,
    diagnostics: NRDiagnostics | None,
    history: Sequence[AcceptedManeuver],
    failures: Sequence[FailureFeedback],
    config: OrchestratorConfig,
) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=_system_prompt(config)),
        ChatMessage(role="user", content=_user_prompt(card, diagnostics, history, failures)),
    ]


def _system_prompt(config: OrchestratorConfig) -> str:
    prompt = f"""You are the Case-2 single grid-resolution agent. Propose exactly one Maneuver.

First reconstruct the relevant grid structure from the Scenario Card before choosing the maneuver.
The orchestrator will apply the maneuver, run the locked AC power flow, and decide convergence.
Do not propose topology switching, load changes, adding/removing components, or more than one action.

{ACTION_VOCABULARY_AND_BOUNDS}

{CAUSE_TAXONOMY}

{CAUSE_ACTION_PRIOR}

{COMPOSITION_AND_PROGRESS}

Diagnostic tools available before the final maneuver: rank_candidate_maneuvers, get_grid_topology, get_action_applicability, and run_ac_pf.
Start each iteration with rank_candidate_maneuvers: it evaluates a spread of legal maneuvers with real power flows and tells you which ones remove overstress. Propose its top candidate unless you have a specific reason not to; use run_ac_pf only to check something the ranking did not cover.
run_ac_pf is your verification tool: pass a candidate maneuver and read 'converged' to see if it fixes the divergence. Test promising candidates with it before proposing; calling it with no maneuver only re-previews the known-diverging base case.
Use at most {MAX_DIAGNOSTIC_TOOL_CALLS} diagnostic tool calls. The final answer must be a propose_maneuver tool call.
Use only the declared atomic Q-V actions.
MANEUVER_BUDGET={config.MANEUVER_BUDGET}; this call proposes one maneuver for the current iteration."""
    return prompt


def _user_prompt(
    card: str,
    diagnostics: NRDiagnostics | None,
    history: Sequence[AcceptedManeuver],
    failures: Sequence[FailureFeedback],
) -> str:
    sections = [
        "# Scenario Card",
        card,
        "# Latest NR diagnostics",
        _diagnostics_text(diagnostics),
    ]
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
