# ABOUTME: Guards the Bedrock Converse contract on every tool schema the agents send.
# ABOUTME: Bedrock rejects a toolSpec whose inputSchema is not a top-level JSON object.
from restorebench.agents import executor, multi_agent, single_agent
from restorebench.agents.tool_loop import MAX_DIAGNOSTIC_TOOL_CALLS, build_tool_config, default_diagnostic_tools


def _every_tool_config() -> list[dict]:
    """Every toolConfig any agent actually sends to Bedrock."""
    return [
        build_tool_config(single_agent._single_agent_diagnostic_tools(), single_agent._maneuver_terminal_tool()),
        build_tool_config(default_diagnostic_tools(), executor._maneuver_terminal_tool()),
        multi_agent._orchestrator_tool_config(),
    ]


def test_every_tool_input_schema_is_a_top_level_object() -> None:
    """Bedrock Converse rejects a toolSpec whose inputSchema.json.type is not "object".

    A Pydantic discriminated union (e.g. Action) serializes to a bare oneOf with no
    top-level "type" — Bedrock answers with a ValidationException and the whole run
    dies with LLM_FAILURE before a single token is generated. Unit tests that fake
    llm_call never send the schema, so only a live call catches it; this guards it
    offline.
    """
    for config in _every_tool_config():
        for tool in config["tools"]:
            spec = tool["toolSpec"]
            schema = spec["inputSchema"]["json"]
            assert schema.get("type") == "object", (
                f"tool {spec['name']!r} has inputSchema type={schema.get('type')!r} "
                f"(keys: {sorted(schema)}) — Bedrock requires a top-level object"
            )


def test_no_tool_input_schema_uses_a_top_level_combinator() -> None:
    """Bedrock: "input_schema does not support oneOf, allOf, or anyOf at the top level".

    This is why the Action union cannot be sent flat and must be wrapped in an object.
    """
    for config in _every_tool_config():
        for tool in config["tools"]:
            spec = tool["toolSpec"]
            schema = spec["inputSchema"]["json"]
            forbidden = {"oneOf", "allOf", "anyOf"} & set(schema)
            assert not forbidden, f"tool {spec['name']!r} has top-level {sorted(forbidden)} — Bedrock rejects it"


def test_action_applicability_wraps_the_action_union_in_an_object() -> None:
    tools = {tool.name: tool for tool in default_diagnostic_tools()}
    schema = tools["get_action_applicability"].input_schema

    assert schema["type"] == "object"
    assert "action" in schema["properties"]
    assert schema["required"] == ["action"]


def test_run_ac_pf_description_states_it_verifies_a_candidate_maneuver() -> None:
    """The model must learn from the tool description that run_ac_pf is the verification tool.

    A live transcript showed the model calling run_ac_pf with no maneuver — a useless re-preview of
    the known-diverging base case — because the old description ("optionally after one Maneuver")
    made the maneuver read as secondary. The description has to say: pass the candidate, it reports
    whether the grid converges, use it before proposing.
    """
    tools = {tool.name: tool for tool in default_diagnostic_tools()}
    desc = tools["run_ac_pf"].description.lower()

    assert "maneuver" in desc
    assert "converg" in desc
    assert "before" in desc  # "test candidates before proposing"


def test_diagnostic_tool_budget_leaves_room_to_verify_candidates() -> None:
    """Three calls is too few: one topology probe plus a single verification exhausts it.

    With ~100 possible actions the agent needs to test several candidates with run_ac_pf, so the
    per-iteration budget must leave real room for verification.
    """
    assert MAX_DIAGNOSTIC_TOOL_CALLS >= 6
