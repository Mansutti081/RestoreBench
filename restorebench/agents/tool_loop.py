# ABOUTME: Provides the reusable Bedrock tool-use loop for diagnostic agent roles.
# ABOUTME: Keeps deterministic grid tools outside role prompts while preserving Converse continuations.
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, model_validator

from restorebench.agents.progress import overstress_verdict
from restorebench.llm.providers import ChatMessage, LLMResponse, ToolResult, ToolUse, llm_call
from restorebench.schemas.actions import Action, Maneuver
from restorebench.schemas.errors import InvalidActionError, LLMFailureError, attach_llm_responses
from restorebench.schemas.feedback import FailedManeuverAttempt, SandboxNet
from restorebench.schemas.power_flow import PowerFlowResult
from restorebench.schemas.topology import ApplicabilityResult
from restorebench.tools.candidate_ranking import rank_candidates, ranking_payload
from restorebench.tools.power_flow import run_ac_pf
from restorebench.tools.sandbox import apply_maneuver, create_sandbox, discard_sandbox, resolve_net
from restorebench.tools.topology import get_action_applicability, get_grid_topology


# Per-iteration diagnostic tool budget. At 3, a topology probe plus one power-flow preview exhausted
# it before the agent could verify a single candidate maneuver, so it proposed blind. The agent must
# be able to test several candidates with run_ac_pf before committing.
MAX_DIAGNOSTIC_TOOL_CALLS = 8
GET_GRID_TOPOLOGY_TOOL_NAME = "get_grid_topology"
GET_ACTION_APPLICABILITY_TOOL_NAME = "get_action_applicability"
RUN_AC_PF_TOOL_NAME = "run_ac_pf"
RANK_CANDIDATE_MANEUVERS_TOOL_NAME = "rank_candidate_maneuvers"
DiagnosticHandler = Callable[[SandboxNet, dict[str, Any]], dict[str, Any]]
LLMCall = Callable[..., LLMResponse]
_ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)


@dataclass(frozen=True)
class DiagnosticTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: DiagnosticHandler


@dataclass(frozen=True)
class TerminalTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_model: type[BaseModel]


@dataclass(frozen=True)
class ToolLoopResult:
    tool_use: ToolUse
    responses: tuple[LLMResponse, ...]


class RunAcPfInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    maneuver: Maneuver | None = None


class ActionApplicabilityInput(BaseModel):
    """Wraps the Action union in an object.

    Bedrock Converse rejects a toolSpec whose `inputSchema.json.type` is not "object", and it
    rejects `oneOf`/`anyOf`/`allOf` at the top level outright ("input_schema does not support
    oneOf, allOf, or anyOf at the top level"). A Pydantic discriminated union is exactly that,
    so the union cannot be sent flat: it has to travel inside an object property.
    """

    model_config = ConfigDict(extra="forbid")

    action: Action

    @model_validator(mode="before")
    @classmethod
    def unwrap_echoed_wrapper(cls, data: Any) -> Any:
        """Unwrap `{"action": {"action": {...}}}`, which models emit for a union-valued property.

        The action inside is well-formed and in-vocabulary; only the envelope is doubled. Every
        field bound still validates below, so tolerating this decodes the transport shape without
        widening the action space.
        """
        if not isinstance(data, dict):
            return data
        inner = data.get("action")
        if isinstance(inner, dict) and "type" not in inner and isinstance(inner.get("action"), dict):
            return {**data, "action": inner["action"]}
        return data


def run_tool_loop(
    *,
    model_id: str,
    messages: list[ChatMessage],
    grid: SandboxNet,
    diagnostic_tools: Sequence[DiagnosticTool],
    terminal_tool: TerminalTool,
    max_diagnostic_tool_calls: int,
    role: str,
    llm_call_fn: LLMCall = llm_call,
) -> ToolLoopResult:
    responses: list[LLMResponse] = []
    n_diagnostic_calls = 0
    must_propose = False
    tools = build_tool_config(diagnostic_tools, terminal_tool)

    # A truncation/throttle from llm_call or a missing-terminal validation raises mid-loop; carry the
    # responses already produced so the orchestrator can still bill the tokens they cost.
    try:
        while True:
            response = llm_call_fn(model_id, messages, temperature=1.0, thinking=True, tools=tools)
            response.raw["role"] = role
            responses.append(response)

            proposal = _proposal_tool(response.tool_uses, terminal_tool.name)
            if proposal is not None:
                return ToolLoopResult(tool_use=proposal, responses=tuple(responses))
            if must_propose:
                _raise_missing_terminal(terminal_tool)

            tool_results, n_executed = _execute_diagnostic_tools(
                response.tool_uses,
                grid,
                n_diagnostic_calls,
                diagnostic_tools,
                terminal_tool.name,
                max_diagnostic_tool_calls,
            )
            n_diagnostic_calls += n_executed
            if not tool_results:
                _raise_missing_terminal(terminal_tool)

            must_propose = n_diagnostic_calls >= max_diagnostic_tool_calls
            messages.extend(_continuation_messages(response, tool_results, must_propose, terminal_tool.name))
    except (LLMFailureError, ValidationError) as exc:
        attach_llm_responses(exc, responses)
        raise


def build_tool_config(
    diagnostic_tools: Sequence[DiagnosticTool],
    terminal_tool: TerminalTool,
) -> dict[str, Any]:
    return {
        "tools": [
            *[
                {
                    "toolSpec": {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": {"json": tool.input_schema},
                    }
                }
                for tool in diagnostic_tools
            ],
            {
                "toolSpec": {
                    "name": terminal_tool.name,
                    "description": terminal_tool.description,
                    "inputSchema": {"json": terminal_tool.input_schema},
                }
            },
        ]
    }


def default_diagnostic_tools(
    saturated_gens: frozenset[int] | None = None,
    *,
    failed_actions: Sequence[Action] = (),
    failed_attempts: list[FailedManeuverAttempt] | None = None,
) -> tuple[DiagnosticTool, ...]:
    blocked_actions = list(failed_actions)
    attempt_sink = failed_attempts if failed_attempts is not None else []
    return (
        DiagnosticTool(
            name=GET_GRID_TOPOLOGY_TOOL_NAME,
            description="Return the current grid topology and controllable element summary.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            handler=_make_grid_topology_handler(saturated_gens),
        ),
        DiagnosticTool(
            name=GET_ACTION_APPLICABILITY_TOOL_NAME,
            description="Check whether one proposed Action is applicable on the diagnostic grid.",
            input_schema=ActionApplicabilityInput.model_json_schema(),
            handler=_make_applicability_handler(saturated_gens or frozenset(), blocked_actions),
        ),
        DiagnosticTool(
            name=RUN_AC_PF_TOOL_NAME,
            description=(
                "Test whether a candidate maneuver restores convergence: pass the maneuver and read "
                "'converged' in the result. This is the ONLY way to know if a maneuver actually fixes "
                "the divergence, so verify promising candidates with it before you propose. When it "
                "does not converge, 'overstress_verdict' says whether the candidate moved the case "
                "closer to solvable ('improved') or further away ('worsened'), with the two readings "
                "in 'overstress_before' and 'overstress_after'. Do not compare these numbers yourself: "
                "an 'improved' candidate is real progress and is worth proposing even though it still "
                "diverges. Calling it without a maneuver only re-previews the known-diverging base "
                "case and wastes a call."
            ),
            input_schema=RunAcPfInput.model_json_schema(),
            handler=_make_run_ac_pf_handler(
                saturated_gens or frozenset(),
                blocked_actions,
                attempt_sink,
            ),
        ),
        DiagnosticTool(
            name=RANK_CANDIDATE_MANEUVERS_TOOL_NAME,
            description=(
                "Survey the grid: evaluates a spread of legal maneuvers with real power flows and "
                "returns them ordered by how much overstress each one removes. 'solves_the_case' is true "
                "only for a candidate that both converges and satisfies the feasibility limits; one "
                "that converges but violates them would waste a maneuver, and is ranked last. Call "
                "this FIRST each "
                "iteration to see where progress is available, then propose the top candidate. When "
                "'any_candidate_converges' is false no single action solves the case and you must "
                "compose: take the best candidate as this iteration's maneuver and the next "
                "iteration will rank again from the improved grid. Every listed candidate is legal "
                "and applicable on the current grid, and has already been evaluated with a real "
                "power flow — re-checking one with run_ac_pf spends a call to learn nothing. "
                "Proposing a maneuver that is not on the list risks INVALID_ACTION, which costs a "
                "maneuver from your budget and leaves the grid unchanged. The shortlist is a sample "
                "of 'n_legal_actions_available', so when nothing in it improves the case, other "
                "legal maneuvers may still help — that is when run_ac_pf earns its call. Costs one "
                "call and about half a minute; calling it again in the same iteration returns the "
                "same answer."
            ),
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            handler=_make_ranking_handler(saturated_gens or frozenset(), blocked_actions),
        ),
    )


def _proposal_tool(tool_uses: Sequence[ToolUse], terminal_tool_name: str) -> ToolUse | None:
    return next((tool_use for tool_use in tool_uses if tool_use.name == terminal_tool_name), None)


def _execute_diagnostic_tools(
    tool_uses: Sequence[ToolUse],
    grid: SandboxNet,
    calls_used: int,
    diagnostic_tools: Sequence[DiagnosticTool],
    terminal_tool_name: str,
    max_diagnostic_tool_calls: int,
) -> tuple[list[dict[str, Any]], int]:
    results: list[dict[str, Any]] = []
    executed = 0
    tools_by_name = {tool.name: tool for tool in diagnostic_tools}
    for tool_use in tool_uses:
        if tool_use.name == terminal_tool_name:
            continue
        if calls_used + executed >= max_diagnostic_tool_calls:
            results.append(_tool_error(tool_use, f"diagnostic tool budget exhausted; call {terminal_tool_name} now"))
            continue
        results.append(_run_diagnostic_tool(tools_by_name, tool_use, grid))
        executed += 1
    return results, executed


def _run_diagnostic_tool(
    tools_by_name: dict[str, DiagnosticTool],
    tool_use: ToolUse,
    grid: SandboxNet,
) -> dict[str, Any]:
    tool = tools_by_name.get(tool_use.name)
    if tool is None:
        return _tool_error(tool_use, f"unknown diagnostic tool {tool_use.name}")
    try:
        return _tool_success(tool_use, tool.handler(grid, tool_use.input))
    except InvalidActionError as exc:
        return _tool_error(tool_use, str(exc))
    except ValidationError as exc:
        # A tool argument the model got wrong is feedback, not a fatal error: handing it back
        # lets the model correct itself. Letting it escape aborts the scenario with LLM_FAILURE
        # and the model never sees what it did wrong.
        return _tool_error(tool_use, f"invalid arguments for {tool_use.name}: {exc}")


def _make_grid_topology_handler(
    saturated_gens: frozenset[int] | None,
) -> DiagnosticHandler:
    def handler(grid: SandboxNet, _tool_input: dict[str, Any]) -> dict[str, Any]:
        return _dump(get_grid_topology(grid, saturated_gens=saturated_gens))

    return handler


def _make_applicability_handler(
    saturated_gens: frozenset[int],
    blocked_actions: list[Action],
) -> DiagnosticHandler:
    # Bind the Q-saturated generators so the check can reject a setpoint raise the solver would discard.
    def handler(grid: SandboxNet, tool_input: dict[str, Any]) -> dict[str, Any]:
        parsed = ActionApplicabilityInput.model_validate(tool_input)
        if _contains_action(blocked_actions, parsed.action):
            return _dump(
                ApplicabilityResult(
                    action=parsed.action,
                    applicable=False,
                    reason="the same action already failed on the current grid state",
                )
            )
        return _dump(get_action_applicability(grid, parsed.action, saturated_gens=saturated_gens))

    return handler


def _make_run_ac_pf_handler(
    saturated_gens: frozenset[int],
    blocked_actions: list[Action],
    failed_attempts: list[FailedManeuverAttempt],
) -> DiagnosticHandler:
    # The grid is fixed for the whole iteration these tools are built for, so its overstress is
    # measured once and reused. Re-solving it per preview would roughly double the wall clock,
    # and every preview needs it: a bare "overstress: 0.024" is unreadable without the value it
    # is being compared against, and a model that guesses the comparison gets it backwards.
    baseline: list[PowerFlowResult] = []

    def baseline_result(grid: SandboxNet) -> PowerFlowResult:
        if not baseline:
            baseline.append(run_ac_pf(grid))
        return baseline[0]

    def handler(grid: SandboxNet, tool_input: dict[str, Any]) -> dict[str, Any]:
        parsed = RunAcPfInput.model_validate(tool_input)
        if parsed.maneuver is None:
            return _compact_power_flow(baseline_result(grid))
        if _contains_action(blocked_actions, parsed.maneuver.action):
            raise InvalidActionError(
                parsed.maneuver.action,
                "the same action already failed on the current grid state; choose a different action",
            )

        # Measured before any sandbox work, so the reading the preview is compared against is
        # always the untouched grid the agent is looking at.
        before = _overstress_of(baseline_result(grid))
        sandbox = create_sandbox(resolve_net(grid))
        try:
            try:
                apply_maneuver(sandbox, parsed.maneuver, saturated_gens=saturated_gens)
            except InvalidActionError as exc:
                _record_failed_attempt(
                    blocked_actions,
                    failed_attempts,
                    FailedManeuverAttempt(
                        kind="PREVIEW_INVALID",
                        maneuver=parsed.maneuver,
                        diagnostics=None,
                        detail=str(exc),
                    ),
                )
                raise
            result = run_ac_pf(sandbox)
            # A preview that closed part of the gap is a candidate, not a failure. Blocking it
            # here forbade the agent from ever proposing the best move it had found: the repeat
            # guard exists to stop it retrying what did not work, and a maneuver that lowers
            # overstress is precisely what did work, short of converging outright.
            if not result.converged and overstress_verdict(before, _overstress_of(result)) != "improved":
                _record_failed_attempt(
                    blocked_actions,
                    failed_attempts,
                    FailedManeuverAttempt(
                        kind="PREVIEW_DIVERGED",
                        maneuver=parsed.maneuver,
                        diagnostics=result.diagnostics,
                        detail=result.error_message,
                    ),
                )
            return _compact_power_flow(result, before=before)
        finally:
            discard_sandbox(sandbox)

    return handler


def _make_ranking_handler(
    saturated_gens: frozenset[int],
    blocked_actions: list[Action],
) -> DiagnosticHandler:
    # Deterministic on an unchanged grid, and the grid is fixed for the iteration these tools
    # were built for. Without this, an agent that calls it twice spends a dozen solves to be
    # told the same thing, and can burn its whole diagnostic budget on one answer.
    cached: list[dict[str, Any]] = []

    def handler(grid: SandboxNet, _tool_input: dict[str, Any]) -> dict[str, Any]:
        if cached:
            return cached[0]
        # The saturated set is the same Q context the applicability guard enforces, so the
        # shortlist can never offer a maneuver the runner would then refuse to apply.
        ranking = rank_candidates(
            resolve_net(grid),
            q_context={gen_id: "Q_LIMITED_UPPER" for gen_id in saturated_gens},
            blocked_actions=list(blocked_actions),
        )
        cached.append(ranking_payload(ranking))
        return cached[0]

    return handler


def _contains_action(actions: Sequence[Action], candidate: Action) -> bool:
    return any(action == candidate for action in actions)


def _record_failed_attempt(
    blocked_actions: list[Action],
    failed_attempts: list[FailedManeuverAttempt],
    attempt: FailedManeuverAttempt,
) -> None:
    if _contains_action(blocked_actions, attempt.maneuver.action):
        return
    blocked_actions.append(attempt.maneuver.action)
    failed_attempts.append(attempt)


def _overstress_of(result: PowerFlowResult) -> float | None:
    return None if result.diagnostics is None else result.diagnostics.overstress


def _compact_power_flow(result: PowerFlowResult, *, before: float | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "converged": result.converged,
        "iterations": result.iterations,
        "tolerance_used": result.tolerance_used,
    }
    if result.converged:
        # Convergence is the terminal signal; a progress verdict alongside it would compete
        # with the only answer that ends the search.
        return payload
    if result.diagnostics is not None:
        payload["diagnostics"] = result.diagnostics.model_dump(mode="json")
    after = _overstress_of(result)
    verdict = overstress_verdict(before, after)
    if verdict != "unknown":
        payload["overstress_before"] = before
        payload["overstress_after"] = after
        payload["overstress_verdict"] = verdict
    return payload


def _tool_success(tool_use: ToolUse, payload: dict[str, Any]) -> dict[str, Any]:
    result = ToolResult(toolUseId=tool_use.tool_use_id, content=[{"json": payload}], status="success")
    return {"toolResult": result.model_dump(mode="json", by_alias=True)}


def _tool_error(tool_use: ToolUse, detail: str) -> dict[str, Any]:
    result = ToolResult(toolUseId=tool_use.tool_use_id, content=[{"text": detail}], status="error")
    return {"toolResult": result.model_dump(mode="json", by_alias=True)}


def retry_messages(response: LLMResponse, *, detail: str, tool_use: ToolUse | None) -> list[ChatMessage]:
    """Build the two messages that ask a model to retry a proposal that failed validation.

    The assistant turn carries the tool_use that produced the bad payload, and the provider
    requires the next message to answer it with a matching tool_result. Replying with plain text
    makes the provider reject the entire request, so the retry never runs and the scenario is
    recorded as a model failure. Every role that retries must build its continuation here: the
    same mistake was written independently in four places before this existed.
    """
    if tool_use is None:
        # Nothing to answer: the assistant turn carries no tool call, so plain text is legal.
        return [
            ChatMessage(role="assistant", content=list(response.assistant_content)),
            ChatMessage(role="user", content=detail),
        ]
    return [
        ChatMessage(role="assistant", content=list(response.assistant_content)),
        ChatMessage(role="user", content=[_tool_error(tool_use, detail)]),
    ]


def _continuation_messages(
    response: LLMResponse,
    tool_results: list[dict[str, Any]],
    must_propose: bool,
    terminal_tool_name: str,
) -> list[ChatMessage]:
    user_content = list(tool_results)
    if must_propose:
        user_content.append({"text": f"Diagnostic tool budget is exhausted. You must now call {terminal_tool_name}."})
    return [
        ChatMessage(role="assistant", content=list(response.assistant_content)),
        ChatMessage(role="user", content=user_content),
    ]


def _dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return {"value": value}


def _raise_missing_terminal(terminal_tool: TerminalTool) -> NoReturn:
    terminal_tool.output_model.model_validate({})
    raise AssertionError("unreachable")
