# ABOUTME: Verifies TopologyServer table extraction and soft action applicability.
# ABOUTME: Guards read-only behavior and Scenario Card consistency for augmented IEEE 118.
from __future__ import annotations

from uuid import uuid4

import pandas as pd
import pandapower as pp

from restorebench.schemas.actions import (
    GenVoltageSetpointAction,
    ShuntStepAction,
    TapAdjustmentAction,
)
from restorebench.schemas.feedback import SandboxNet
from restorebench.schemas.topology import TopologySummary
from restorebench.tools import topology as topo
from restorebench.corpus.augment import build_augmented_base
from restorebench.corpus import render_scenario_card as render


GRID_TABLES = ("bus", "line", "trafo", "gen", "ext_grid", "load", "shunt")


def _dispatchable_gen_id(net) -> int:
    mask = net.gen["in_service"].astype(bool) & (net.gen["min_p_mw"] < net.gen["max_p_mw"])
    return int(net.gen.index[mask][0])


def _tappable_trafo_id(net) -> int:
    mask = net.trafo["in_service"].astype(bool) & net.trafo["tap_pos"].notna()
    return int(net.trafo.index[mask][0])


def _nontappable_trafo_id(net) -> int:
    mask = net.trafo["tap_pos"].isna()
    return int(net.trafo.index[mask][0])


def _next_tap_in_range(net, trafo_id: int) -> int:
    current = int(net.trafo.at[trafo_id, "tap_pos"])
    tap_min = int(net.trafo.at[trafo_id, "tap_min"])
    tap_max = int(net.trafo.at[trafo_id, "tap_max"])
    if current < tap_max:
        return current + 1
    return current - 1 if current > tap_min else current


def _next_vm_in_range(net, gen_id: int) -> float:
    current = float(net.gen.at[gen_id, "vm_pu"])
    return round(current + 0.01, 10) if current < 1.05 else round(current - 0.01, 10)


def test_get_grid_topology_extracts_augmented_base_tables_without_power_flow(monkeypatch):
    net = build_augmented_base()

    def fail_runpp(*args, **kwargs) -> None:
        raise AssertionError("TopologyServer must not run a power flow")

    monkeypatch.setattr(pp, "runpp", fail_runpp)

    summary = topo.get_grid_topology(net)

    assert summary.n_buses == len(net.bus)
    assert summary.n_lines == len(net.line)
    assert summary.n_trafos == len(net.trafo)
    assert summary.n_gens == len(net.gen)
    assert summary.n_ext_grids == len(net.ext_grid)
    assert summary.n_loads == len(net.load)
    assert summary.n_shunts == len(net.shunt)
    assert summary.slack_bus == int(net.ext_grid.bus.iloc[0])

    assert [bus.bus_id for bus in summary.buses] == [int(idx) for idx in net.bus.sort_index().index]
    assert [line.line_id for line in summary.lines] == [int(idx) for idx in net.line.sort_index().index]

    dispatchable_by_id = {gen.gen_id: gen.dispatchable for gen in summary.gens}
    voltage_status_by_id = {gen.gen_id: gen.voltage_control_status for gen in summary.gens}
    for gen_id, row in net.gen.sort_index().iterrows():
        assert dispatchable_by_id[int(gen_id)] is (float(row["min_p_mw"]) < float(row["max_p_mw"]))
        assert voltage_status_by_id[int(gen_id)] is None

    condenser_ids = [int(idx) for idx in net.gen.index[net.gen["min_p_mw"] == net.gen["max_p_mw"]]]
    assert condenser_ids
    assert all(dispatchable_by_id[gen_id] is False for gen_id in condenser_ids)

    shunt_q_by_id = {shunt.shunt_id: shunt.q_mvar for shunt in summary.shunts}
    for shunt_id, row in net.shunt.sort_index().iterrows():
        assert shunt_q_by_id[int(shunt_id)] == float(row["q_mvar"])
        assert (shunt_q_by_id[int(shunt_id)] < 0) is (float(row["q_mvar"]) < 0)

    assert TopologySummary.model_validate(summary.model_dump()) == summary
    assert TopologySummary.model_validate_json(summary.model_dump_json()) == summary


def test_get_grid_topology_includes_out_of_service_elements_with_flags():
    net = build_augmented_base()
    line_id = int(net.line.index[0])
    gen_id = int(net.gen.index[0])
    trafo_id = _tappable_trafo_id(net)
    shunt_id = int(net.shunt.index[0])

    net.line.at[line_id, "in_service"] = False
    net.gen.at[gen_id, "in_service"] = False
    net.trafo.at[trafo_id, "in_service"] = False
    net.shunt.at[shunt_id, "in_service"] = False

    summary = topo.get_grid_topology(net)

    assert next(line for line in summary.lines if line.line_id == line_id).in_service is False
    assert next(gen for gen in summary.gens if gen.gen_id == gen_id).in_service is False
    assert next(trafo for trafo in summary.trafos if trafo.trafo_id == trafo_id).in_service is False
    assert next(shunt for shunt in summary.shunts if shunt.shunt_id == shunt_id).in_service is False
    assert summary.n_lines == len(net.line)
    assert summary.n_gens == len(net.gen)
    assert summary.n_trafos == len(net.trafo)
    assert summary.n_shunts == len(net.shunt)


def test_topology_summary_matches_scenario_card_action_lever_tables():
    net = build_augmented_base()
    summary = topo.get_grid_topology(net)
    card = render.render_scenario_card(net)

    for gen in summary.gens:
        if gen.in_service:
            control_row = (
                f"| {gen.gen_id} | {gen.bus} | true | {gen.vm_pu:.3f} | "
                "0.950 | 1.050 | 0.010 |"
            )
            dispatch_row = (
                f"| {gen.gen_id} | {gen.bus} | true | {gen.p_mw:.1f} | "
                f"{gen.min_p_mw:.1f} | {gen.max_p_mw:.1f} |"
            )
            assert control_row in card
            assert dispatch_row in card

    tappable_summary_ids = {
        trafo.trafo_id for trafo in summary.trafos if trafo.in_service and trafo.tap_pos is not None
    }
    tappable_card_ids = set(
        int(idx) for idx in net.trafo.index[net.trafo["in_service"].astype(bool) & net.trafo["tap_pos"].notna()]
    )
    assert tappable_summary_ids == tappable_card_ids
    for trafo in summary.trafos:
        if trafo.trafo_id in tappable_card_ids:
            expected = (
                f"| {trafo.trafo_id} | {trafo.hv_bus} | {trafo.lv_bus} | true | "
            )
            assert expected in card

    for shunt in summary.shunts:
        state = "true" if shunt.in_service else "false"
        shunt_type = "capacitor" if shunt.q_mvar < 0 else "reactor"
        expected = (
            f"| {shunt.shunt_id} | {shunt.bus} | {state} | "
            f"{shunt.q_mvar:.1f} | {shunt_type} |"
        )
        assert expected in card


def test_raising_the_setpoint_of_a_q_saturated_generator_is_not_applicable():
    """A generator pinned at its Q max has switched to PQ; raising its voltage setpoint cannot inject
    more reactive power, so the move is a no-op. The applicability check must reject it (direction-aware:
    lowering the setpoint de-saturates it and is still allowed).
    """
    net = build_augmented_base()
    gen_id = _dispatchable_gen_id(net)
    net.gen.at[gen_id, "vm_pu"] = 1.0  # mid-band so both raise and lower stay inside the [0.95, 1.05] schema bounds
    saturated = frozenset({gen_id})

    raise_setpoint = GenVoltageSetpointAction(type="GEN_V_SETPOINT", gen_id=gen_id, new_vm_pu=1.01)
    lower_setpoint = GenVoltageSetpointAction(type="GEN_V_SETPOINT", gen_id=gen_id, new_vm_pu=0.99)

    # Saturated + raise → rejected with a reason that names the saturation.
    raised = topo.get_action_applicability(net, raise_setpoint, saturated_gens=saturated)
    assert raised.applicable is False
    assert raised.reason and "saturat" in raised.reason.lower()

    # Saturated + lower → still applicable (it de-saturates the machine).
    assert topo.get_action_applicability(net, lower_setpoint, saturated_gens=saturated).applicable is True

    # Not saturated + raise → applicable (default: no saturation set).
    assert topo.get_action_applicability(net, raise_setpoint).applicable is True


def test_action_applicability_screens_generator_actions():
    net = build_augmented_base()
    gen_id = _dispatchable_gen_id(net)

    voltage = GenVoltageSetpointAction(
        type="GEN_V_SETPOINT",
        gen_id=gen_id,
        new_vm_pu=_next_vm_in_range(net, gen_id),
    )
    assert topo.get_action_applicability(net, voltage).applicable is True

    net.gen.at[gen_id, "in_service"] = False
    out_of_service_voltage = topo.get_action_applicability(net, voltage)
    assert out_of_service_voltage.applicable is False
    assert out_of_service_voltage.reason
    net.gen.at[gen_id, "in_service"] = True

    non_atomic = GenVoltageSetpointAction(
        type="GEN_V_SETPOINT",
        gen_id=gen_id,
        new_vm_pu=round(float(net.gen.at[gen_id, "vm_pu"]) - 0.02, 10),
    )
    non_atomic_result = topo.get_action_applicability(net, non_atomic)
    assert non_atomic_result.applicable is False
    assert non_atomic_result.reason and "atomic" in non_atomic_result.reason


def test_action_applicability_screens_shunt_and_tap_actions():
    net = build_augmented_base()
    shunt_id = int(net.shunt.index[0])
    current_shunt_step = int(net.shunt.at[shunt_id, "step"])
    tap_id = _tappable_trafo_id(net)

    no_op_shunt = ShuntStepAction(
        type="SHUNT_STEP",
        shunt_id=shunt_id,
        new_step=current_shunt_step,
    )
    no_op_result = topo.get_action_applicability(net, no_op_shunt)
    assert no_op_result.applicable is False
    assert no_op_result.reason and "no-op" in no_op_result.reason

    toggle_shunt = ShuntStepAction(
        type="SHUNT_STEP",
        shunt_id=shunt_id,
        new_step=1 - current_shunt_step,
    )
    assert topo.get_action_applicability(net, toggle_shunt).applicable is True

    valid_tap = TapAdjustmentAction(
        type="TAP_ADJUSTMENT",
        trafo_id=tap_id,
        new_tap_pos=_next_tap_in_range(net, tap_id),
    )
    assert topo.get_action_applicability(net, valid_tap).applicable is True

    no_op_tap = TapAdjustmentAction(
        type="TAP_ADJUSTMENT",
        trafo_id=tap_id,
        new_tap_pos=int(net.trafo.at[tap_id, "tap_pos"]),
    )
    no_op_tap_result = topo.get_action_applicability(net, no_op_tap)
    assert no_op_tap_result.applicable is False
    assert no_op_tap_result.reason and "no-op" in no_op_tap_result.reason

    net.trafo.at[tap_id, "tap_min"] = -1
    net.trafo.at[tap_id, "tap_max"] = 1
    outside_tap = TapAdjustmentAction(type="TAP_ADJUSTMENT", trafo_id=tap_id, new_tap_pos=2)
    outside_tap_result = topo.get_action_applicability(net, outside_tap)
    assert outside_tap_result.applicable is False
    assert outside_tap_result.reason

    nontappable_tap = TapAdjustmentAction(
        type="TAP_ADJUSTMENT",
        trafo_id=_nontappable_trafo_id(net),
        new_tap_pos=0,
    )
    nontappable_result = topo.get_action_applicability(net, nontappable_tap)
    assert nontappable_result.applicable is False
    assert nontappable_result.reason and "tappable" in nontappable_result.reason


def test_topology_tools_accept_sandbox_handle_via_resolve_net(monkeypatch):
    net = build_augmented_base()
    handle = SandboxNet(sandbox_id=uuid4(), scenario_request_id=uuid4())
    resolved_values = []

    class StubSandboxServer:
        @staticmethod
        def resolve_net(value):
            resolved_values.append(value)
            return net

    monkeypatch.setattr(topo, "SandboxServer", StubSandboxServer)

    summary = topo.get_grid_topology(handle)
    action = GenVoltageSetpointAction(
        type="GEN_V_SETPOINT",
        gen_id=_dispatchable_gen_id(net),
        new_vm_pu=_next_vm_in_range(net, _dispatchable_gen_id(net)),
    )
    applicability = topo.get_action_applicability(handle, action)

    assert summary.n_buses == len(net.bus)
    assert applicability.applicable is True
    assert resolved_values == [handle, handle]


def test_topology_tools_are_read_only():
    net = build_augmented_base()
    snapshots = {table: getattr(net, table).copy(deep=True) for table in GRID_TABLES}
    gen_id = _dispatchable_gen_id(net)
    action = GenVoltageSetpointAction(
        type="GEN_V_SETPOINT",
        gen_id=gen_id,
        new_vm_pu=_next_vm_in_range(net, gen_id),
    )

    topo.get_grid_topology(net)
    topo.get_action_applicability(net, action)

    for table in GRID_TABLES:
        pd.testing.assert_frame_equal(getattr(net, table), snapshots[table])
