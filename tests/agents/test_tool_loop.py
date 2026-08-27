# ABOUTME: Verifies the reusable diagnostic tool loop shared by agent roles.
# ABOUTME: Uses scripted LLM responses so no Bedrock call is made.
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from restorebench.agents.tool_loop import DiagnosticTool, TerminalTool, run_tool_loop
from restorebench.llm.providers import ChatMessage, LLMResponse, ToolUse
from restorebench.schemas.actions import Maneuver
from restorebench.tools.sandbox import create_sandbox, discard_sandbox


MODEL_ID = "test-model"


def _maneuver_payload() -> dict[str, Any]:
    return {
        "action": {"type": "GEN_V_SETPOINT", "gen_id": 11, "new_vm_pu": 1.05},
        "diagnosed_cause": "REACTIVE_DEFICIT",
        "rationale": "raise generator voltage",
    }


def _tool_response(tool_name: str, tool_input: dict[str, Any], tool_use_id: str) -> LLMResponse:
    tool_use = ToolUse(toolUseId=tool_use_id, name=tool_name, input=tool_input)
    assistant_content = ({"toolUse": tool_use.model_dump(mode="json", by_alias=True)},)
    return LLMResponse(
        text="tool call",
        model_id=MODEL_ID,
        tokens_in=10,
        tokens_out=4,
        latency_seconds=0.01,
        raw={},
        tool_use=tool_use,
        assistant_content=assistant_content,
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


def test_tool_loop_answers_diagnostic_tool_then_returns_terminal_tool() -> None:
    grid = create_sandbox({"diagnostic": "grid"})
    seen: list[tuple[Any, dict[str, Any]]] = []
    fake_llm = ScriptedLLM(
        [
            _tool_response("inspect_grid", {"bus": 44}, "toolu_inspect"),
            _tool_response("finish", _maneuver_payload(), "toolu_finish"),
        ]
    )

    def inspect(grid_arg: Any, tool_input: dict[str, Any]) -> dict[str, Any]:
        seen.append((grid_arg, tool_input))
        return {"seen_bus": tool_input["bus"]}

    try:
        result = run_tool_loop(
            model_id=MODEL_ID,
            messages=[ChatMessage(role="user", content="fix this grid")],
            grid=grid,
            diagnostic_tools=(
                DiagnosticTool(
                    name="inspect_grid",
                    description="Inspect the grid.",
                    input_schema={"type": "object", "properties": {"bus": {"type": "integer"}}},
                    handler=inspect,
                ),
            ),
            terminal_tool=TerminalTool(
                name="finish",
                description="Return the final maneuver.",
                input_schema=Maneuver.model_json_schema(),
                output_model=Maneuver,
            ),
            max_diagnostic_tool_calls=3,
            role="executor",
            llm_call_fn=fake_llm,
        )
    finally:
        discard_sandbox(grid)

    assert result.tool_use.input == _maneuver_payload()
    assert result.responses[0].raw["role"] == "executor"
    assert result.responses[1].raw["role"] == "executor"
    assert seen == [(grid, {"bus": 44})]
    assert fake_llm.requests[0]["thinking"] is True
    assert fake_llm.requests[0]["temperature"] == 1.0
    assert {tool["toolSpec"]["name"] for tool in fake_llm.requests[0]["tools"]["tools"]} == {
        "inspect_grid",
        "finish",
    }
    continuation_messages = fake_llm.requests[1]["messages"]
    assert continuation_messages[-2].content == list(result.responses[0].assistant_content)
    assert continuation_messages[-1].content == [
        {
            "toolResult": {
                "toolUseId": "toolu_inspect",
                "content": [{"json": {"seen_bus": 44}}],
                "status": "success",
            }
        }
    ]
