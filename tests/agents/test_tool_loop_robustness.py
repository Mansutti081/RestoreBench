# ABOUTME: Guards the tool loop against malformed tool inputs the model sends back.
# ABOUTME: A bad diagnostic tool input must return an error result the model can fix, never kill the run.
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from restorebench.agents.tool_loop import DiagnosticTool, TerminalTool, default_diagnostic_tools, run_tool_loop
from restorebench.llm.providers import ChatMessage, LLMResponse, ToolUse
from restorebench.schemas.actions import Maneuver
from restorebench.tools.sandbox import create_sandbox, discard_sandbox


MODEL_ID = "test-model"


def _maneuver_payload() -> dict[str, Any]:
    return {
        "action": {"type": "GEN_V_SETPOINT", "gen_id": 11, "new_vm_pu": 1.05},
        "diagnosed_cause": "REACTIVE_DEFICIT",
        "rationale": "raise reactive support",
    }


def _response(tool_uses: list[ToolUse]) -> LLMResponse:
    return LLMResponse(
        text="",
        model_id=MODEL_ID,
        tool_uses=tuple(tool_uses),
        assistant_content=({"text": "thinking"},),
        tokens_in=10,
        tokens_out=5,
        tokens_total=15,
        latency_seconds=0.1,
        raw={},
    )


class ScriptedLLM:
    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self.responses = list(responses)

    def __call__(self, *_args: Any, **_kwargs: Any) -> LLMResponse:
        return self.responses.pop(0)


def _terminal_tool() -> TerminalTool:
    return TerminalTool(
        name="propose_maneuver",
        description="Return one Maneuver.",
        input_schema=Maneuver.model_json_schema(),
        output_model=Maneuver,
    )


def _run(responses: Sequence[LLMResponse], tools: Sequence[DiagnosticTool] | None = None) -> Any:
    grid = create_sandbox({"diagnostic": "grid"})
    try:
        return run_tool_loop(
            model_id=MODEL_ID,
            messages=[ChatMessage(role="user", content="fix the grid")],
            grid=grid,
            diagnostic_tools=tools if tools is not None else default_diagnostic_tools(),
            terminal_tool=_terminal_tool(),
            max_diagnostic_tool_calls=3,
            role="single_agent",
            llm_call_fn=ScriptedLLM(responses),
        )
    finally:
        discard_sandbox(grid)


def test_a_malformed_diagnostic_tool_input_is_reported_to_the_model_not_raised() -> None:
    """A tool input the model got wrong must come back as a tool error so it can retry.

    Letting the ValidationError escape aborts the whole scenario with LLM_FAILURE — one bad
    tool argument then costs the run, and the model never gets the chance to correct itself.
    """
    responses = [
        _response([ToolUse(tool_use_id="toolu_bad", name="get_action_applicability", input={"action": "not an action"})]),
        _response([ToolUse(tool_use_id="toolu_final", name="propose_maneuver", input=_maneuver_payload())]),
    ]

    result = _run(responses)

    assert result.tool_use.name == "propose_maneuver"
    assert len(result.responses) == 2


def test_an_unparseable_run_ac_pf_input_is_reported_to_the_model_not_raised() -> None:
    responses = [
        _response([ToolUse(tool_use_id="toolu_bad", name="run_ac_pf", input={"maneuver": {"nonsense": True}})]),
        _response([ToolUse(tool_use_id="toolu_final", name="propose_maneuver", input=_maneuver_payload())]),
    ]

    result = _run(responses)

    assert result.tool_use.name == "propose_maneuver"


def test_a_mid_loop_llm_failure_carries_the_responses_already_produced() -> None:
    """When the loop dies mid-flight, the responses it already got must ride out on the exception.

    Otherwise the orchestrator records zero tokens for a call it actually paid for.
    """
    import pytest

    from restorebench.schemas.errors import LLMFailureError, llm_responses_of

    noop_tool = DiagnosticTool(
        name="probe",
        description="grid-free probe",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda _grid, _inp: {"ok": True},
    )
    first = _response([ToolUse(tool_use_id="toolu_1", name="probe", input={})])

    class FailingSecondCall:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, *_args: Any, **_kwargs: Any) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                return first
            raise LLMFailureError(MODEL_ID, "throttled")

    grid = create_sandbox({"diagnostic": "grid"})
    try:
        with pytest.raises(LLMFailureError) as error:
            run_tool_loop(
                model_id=MODEL_ID,
                messages=[ChatMessage(role="user", content="fix the grid")],
                grid=grid,
                diagnostic_tools=(noop_tool,),
                terminal_tool=_terminal_tool(),
                max_diagnostic_tool_calls=3,
                role="single_agent",
                llm_call_fn=FailingSecondCall(),
            )
    finally:
        discard_sandbox(grid)

    assert llm_responses_of(error.value) == (first,)
