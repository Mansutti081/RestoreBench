# ABOUTME: Verifies the Q-saturated-gen guard is wired from diagnostics through to the applicability tool.
# ABOUTME: The agent must be told "not applicable" when it probes a setpoint raise on a saturated generator.
from __future__ import annotations

import pytest

from restorebench.agents.single_agent import _saturated_gens
from restorebench.agents.tool_loop import (
    GET_ACTION_APPLICABILITY_TOOL_NAME,
    GET_GRID_TOPOLOGY_TOOL_NAME,
    RUN_AC_PF_TOOL_NAME,
    default_diagnostic_tools,
)
from restorebench.schemas.errors import InvalidActionError
from restorebench.schemas.power_flow import NRDiagnostics
from restorebench.tools.sandbox import create_sandbox, discard_sandbox
from restorebench.corpus.augment import build_augmented_base


def _diagnostics(gens_at_q_limit: list[int]) -> NRDiagnostics:
    return NRDiagnostics(
        iterations_attempted=30,
        worst_bus=1,
        lowest_vm_pu=0.6,
        lowest_vm_bus=1,
        gens_at_q_limit=gens_at_q_limit,
        error_message="did not converge",
        diagnostics_source="local_nose",
    )


def test_saturated_gens_extracts_the_set_from_diagnostics() -> None:
    assert _saturated_gens(_diagnostics([3, 7, 12])) == frozenset({3, 7, 12})
    assert _saturated_gens(None) == frozenset()


def test_applicability_tool_built_with_saturated_set_rejects_a_setpoint_raise() -> None:
    net = build_augmented_base()
    gen_id = int(net.gen.index[0])
    other_gen = int(net.gen.index[1])
    net.gen.at[gen_id, "vm_pu"] = 1.0
    net.gen.at[other_gen, "vm_pu"] = 1.0
    grid = create_sandbox(net)

    try:
        tools = {tool.name: tool for tool in default_diagnostic_tools(saturated_gens=frozenset({gen_id}))}
        handler = tools[GET_ACTION_APPLICABILITY_TOOL_NAME].handler

        raised = handler(grid, {"action": {"type": "GEN_V_SETPOINT", "gen_id": gen_id, "new_vm_pu": 1.01}})
        assert raised["applicable"] is False
        assert "saturat" in raised["reason"].lower()

        # A generator NOT in the saturated set is still applicable.
        ok = handler(grid, {"action": {"type": "GEN_V_SETPOINT", "gen_id": other_gen, "new_vm_pu": 1.01}})
        assert ok["applicable"] is True
    finally:
        discard_sandbox(grid)


def test_topology_tool_reports_q_saturated_generators_as_upper_limited() -> None:
    net = build_augmented_base()
    gen_id = int(net.gen.index[0])
    grid = create_sandbox(net)

    try:
        tools = {tool.name: tool for tool in default_diagnostic_tools(saturated_gens=frozenset({gen_id}))}
        topology = tools[GET_GRID_TOPOLOGY_TOOL_NAME].handler(grid, {})
    finally:
        discard_sandbox(grid)

    status_by_gen = {gen["gen_id"]: gen["voltage_control_status"] for gen in topology["gens"]}
    assert status_by_gen[gen_id] == "Q_LIMITED_UPPER"
    assert status_by_gen[int(net.gen.index[1])] == "PV_CONTROLLABLE"


def test_run_ac_pf_tool_rejects_raise_for_q_saturated_generator() -> None:
    net = build_augmented_base()
    gen_id = int(net.gen.index[0])
    net.gen.at[gen_id, "vm_pu"] = 1.0
    grid = create_sandbox(net)
    tool_input = {
        "maneuver": {
            "action": {"type": "GEN_V_SETPOINT", "gen_id": gen_id, "new_vm_pu": 1.01},
            "diagnosed_cause": "REACTIVE_DEFICIT",
            "rationale": "probe an invalid saturated-generator raise",
        }
    }

    try:
        tools = {tool.name: tool for tool in default_diagnostic_tools(saturated_gens=frozenset({gen_id}))}
        with pytest.raises(InvalidActionError, match="Q-saturated"):
            tools[RUN_AC_PF_TOOL_NAME].handler(grid, tool_input)
    finally:
        discard_sandbox(grid)
