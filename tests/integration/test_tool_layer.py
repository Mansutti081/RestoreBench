# ABOUTME: Verifies cross-tool wiring for Sandbox, PowerFlow, Topology, Memory, and schemas.
# ABOUTME: Uses real IEEE-118 dataset scenarios with no mocks, LLM calls, or network access.
from __future__ import annotations

import copy
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandapower as pp
import pytest

import restorebench.schemas as schemas
from restorebench.schemas import (
    ExecutionTrace,
    GenVoltageSetpointAction,
    InvalidActionError,
    Maneuver,
    ShuntStepAction,
    TapAdjustmentAction,
    TraceEvent,
)
from restorebench.tools import power_flow as PowerFlowServer
from restorebench.tools import sandbox as SandboxServer
from restorebench.tools import topology as TopologyServer
from restorebench.corpus.augment import build_augmented_base


DATASET_FULL = Path("dataset/ieee118/full")


def load_full_scenario(scenario_id: str):
    return pp.from_json(str(DATASET_FULL / f"{scenario_id}.json"))


@pytest.fixture
def augmented_base():
    return build_augmented_base()


@pytest.fixture
def diverging_scenario():
    return load_full_scenario("S0008")


def net_bytes(net) -> bytes:
    return pickle.dumps(net, protocol=pickle.HIGHEST_PROTOCOL)


def _maneuver(action) -> Maneuver:
    return Maneuver(action=action, diagnosed_cause=None, rationale="integration test maneuver")


def _certified_max_reactive_maneuvers(net, *, tap_position: int = -2) -> list[Maneuver]:
    maneuvers: list[Maneuver] = []
    for gen_id, row in net.gen.loc[net.gen["in_service"].astype(bool)].sort_index().iterrows():
        current_vm = float(row["vm_pu"])
        while current_vm < 1.05 - 1e-10:
            current_vm = round(min(current_vm + 0.01, 1.05), 10)
            maneuvers.append(
                _maneuver(
                    GenVoltageSetpointAction(
                        type="GEN_V_SETPOINT",
                        gen_id=int(gen_id),
                        new_vm_pu=current_vm,
                    )
                )
            )

    for shunt_id, row in net.shunt.sort_index().iterrows():
        if not bool(row["in_service"]):
            continue
        target_step = 1 if float(row["q_mvar"]) < 0.0 else 0
        if int(row["step"]) != target_step:
            maneuvers.append(
                _maneuver(
                    ShuntStepAction(
                        type="SHUNT_STEP",
                        shunt_id=int(shunt_id),
                        new_step=target_step,
                    )
                )
            )

    tappable = net.trafo["in_service"].astype(bool) & net.trafo["tap_pos"].notna()
    for trafo_id, row in net.trafo.loc[tappable].sort_index().iterrows():
        current_tap = int(row["tap_pos"])
        while current_tap != tap_position:
            current_tap += 1 if tap_position > current_tap else -1
            maneuvers.append(
                _maneuver(
                    TapAdjustmentAction(
                        type="TAP_ADJUSTMENT",
                        trafo_id=int(trafo_id),
                        new_tap_pos=current_tap,
                    )
                )
            )
    return maneuvers


def _apply_maneuvers(handle, maneuvers: list[Maneuver]) -> None:
    for maneuver in maneuvers:
        SandboxServer.apply_maneuver(handle, maneuver, saturated_gens=frozenset())


def _apply_certified_controls_directly(net, *, tap_position: int = -2) -> None:
    net.gen.loc[net.gen["in_service"].astype(bool), "vm_pu"] = 1.05
    available_shunts = net.shunt["in_service"].astype(bool)
    net.shunt.loc[available_shunts & (net.shunt["q_mvar"] > 0), "step"] = 0
    net.shunt.loc[available_shunts & (net.shunt["q_mvar"] < 0), "step"] = 1
    tappable = net.trafo["in_service"].astype(bool) & net.trafo["tap_pos"].notna()
    net.trafo.loc[tappable, "tap_pos"] = int(tap_position)


def _locked_pf_verdict_and_net(net):
    solved = copy.deepcopy(net)
    first_error = PowerFlowServer._try_run_locked_pf(solved, PowerFlowServer.DEFAULT_TOLERANCE)
    if first_error is None:
        return True, solved

    second_error = PowerFlowServer._try_run_locked_pf(solved, PowerFlowServer.RECOVERY_TOLERANCE)
    return second_error is None, solved


def _first_dispatchable_gen(net) -> int:
    mask = net.gen["in_service"].astype(bool) & (net.gen["min_p_mw"] < net.gen["max_p_mw"])
    return int(net.gen.index[mask][0])


def _first_non_tappable_trafo(net) -> int:
    mask = net.trafo["in_service"].astype(bool) & net.trafo["tap_pos"].isna()
    return int(net.trafo.index[mask][0])


def _apply_raises_invalid_action(net, maneuver: Maneuver) -> bool:
    handle = SandboxServer.create_sandbox(net)
    try:
        try:
            SandboxServer.apply_maneuver(handle, maneuver, saturated_gens=frozenset())
        except InvalidActionError:
            return True
        return False
    finally:
        SandboxServer.discard_sandbox(handle)


def _first_tappable_trafo_from_topology(topology):
    return next(trafo for trafo in topology.trafos if trafo.in_service and trafo.tap_pos is not None)


def region_map_for(topology) -> dict[int, int]:
    return {bus.bus_id: 0 for bus in topology.buses}


def _trace_with_diagnostics(request_id, diagnostics):
    return ExecutionTrace(
        request_id=request_id,
        events=[
            TraceEvent(
                timestamp=datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc),
                phase="solve",
                event_name="baseline_diagnostics",
                duration_ms=1.0,
                payload={"diagnostics": diagnostics.model_dump()},
            )
        ],
        n_llm_calls=0,
        total_llm_tokens_in=0,
        total_llm_tokens_out=0,
        n_tool_calls=3,
        n_power_flows=2,
    )


def test_a1_real_dataset_scenario_diverges_with_usable_diagnostics(diverging_scenario):
    result = PowerFlowServer.run_ac_pf(diverging_scenario)

    assert result.converged is False
    assert result.quality is None
    assert result.diagnostics is not None
    assert result.diagnostics.diagnostics_source == "local_nose"
    assert np.isfinite(result.diagnostics.lowest_vm_pu)
    assert result.diagnostics.lowest_vm_pu > 0
    assert result.diagnostics.overstress is not None
    assert np.isfinite(result.diagnostics.overstress)
    assert result.diagnostics.overstress > 0
    assert result.diagnostics.gens_at_q_limit


def test_a2_sandbox_handle_path_converges_after_certified_controls(diverging_scenario):
    handle = SandboxServer.create_sandbox(diverging_scenario)
    try:
        _apply_maneuvers(handle, _certified_max_reactive_maneuvers(diverging_scenario, tap_position=-2))

        result = PowerFlowServer.run_ac_pf(handle)

        assert result.converged is True
        assert result.quality is not None
        assert result.diagnostics is None
    finally:
        SandboxServer.discard_sandbox(handle)


def test_a3_sandbox_powerflow_path_preserves_caller_net_byte_identity(diverging_scenario):
    before = net_bytes(diverging_scenario)
    handle = SandboxServer.create_sandbox(diverging_scenario)
    try:
        _apply_maneuvers(handle, _certified_max_reactive_maneuvers(diverging_scenario, tap_position=-2))
        result = PowerFlowServer.run_ac_pf(handle)

        assert result.converged is True
        assert net_bytes(diverging_scenario) == before
    finally:
        SandboxServer.discard_sandbox(handle)


def test_a4_sandbox_powerflow_reproduces_direct_locked_solve(diverging_scenario):
    handle = SandboxServer.create_sandbox(diverging_scenario)
    try:
        _apply_maneuvers(handle, _certified_max_reactive_maneuvers(diverging_scenario, tap_position=-2))
        handle_result = PowerFlowServer.run_ac_pf(handle)

        handle_verdict, handle_solved = _locked_pf_verdict_and_net(SandboxServer.resolve_net(handle))
        direct_net = copy.deepcopy(diverging_scenario)
        _apply_certified_controls_directly(direct_net, tap_position=-2)
        direct_verdict, direct_solved = _locked_pf_verdict_and_net(direct_net)

        assert handle_result.converged is True
        assert handle_verdict == direct_verdict == handle_result.converged
        assert np.allclose(
            handle_solved.res_bus["vm_pu"].to_numpy(),
            direct_solved.res_bus["vm_pu"].to_numpy(),
            atol=1e-6,
        )
    finally:
        SandboxServer.discard_sandbox(handle)


def test_b1_topology_handle_reflects_applied_shunt_maneuver(augmented_base):
    handle = SandboxServer.create_sandbox(augmented_base)
    try:
        before_topology = TopologyServer.get_grid_topology(handle)
        capacitor_id = int(augmented_base.shunt.index[augmented_base.shunt["q_mvar"] < 0][0])
        shunt = next(item for item in before_topology.shunts if item.shunt_id == capacitor_id)
        target_step = 1 - shunt.step

        SandboxServer.apply_maneuver(
            handle,
            _maneuver(
                ShuntStepAction(
                    type="SHUNT_STEP",
                    shunt_id=shunt.shunt_id,
                    new_step=target_step,
                )
            ),
            saturated_gens=frozenset(),
        )
        after_topology = TopologyServer.get_grid_topology(handle)

        updated = next(item for item in after_topology.shunts if item.shunt_id == shunt.shunt_id)
        assert shunt.type == "capacitor"
        assert updated.step == target_step
        assert updated.in_service is shunt.in_service
        assert updated.type == shunt.type
    finally:
        SandboxServer.discard_sandbox(handle)


def test_b2_hard_gate_rejections_match_soft_screen_rejections(augmented_base):
    gen_id = _first_dispatchable_gen(augmented_base)
    non_tappable_trafo_id = _first_non_tappable_trafo(augmented_base)
    shunt_id = int(augmented_base.shunt.index[0])

    out_of_service = copy.deepcopy(augmented_base)
    out_of_service.gen.at[gen_id, "in_service"] = False
    unavailable_shunt = copy.deepcopy(augmented_base)
    unavailable_shunt.shunt.at[shunt_id, "in_service"] = False

    cases = [
        (
            augmented_base,
            _maneuver(
                GenVoltageSetpointAction(
                    type="GEN_V_SETPOINT",
                    gen_id=int(augmented_base.gen.index.max()) + 1000,
                    new_vm_pu=1.0,
                )
            ),
        ),
        (
            out_of_service,
            _maneuver(GenVoltageSetpointAction(type="GEN_V_SETPOINT", gen_id=gen_id, new_vm_pu=1.0)),
        ),
        (
            augmented_base,
            _maneuver(
                TapAdjustmentAction(
                    type="TAP_ADJUSTMENT",
                    trafo_id=non_tappable_trafo_id,
                    new_tap_pos=0,
                )
            ),
        ),
        (
            augmented_base,
            _maneuver(
                GenVoltageSetpointAction(
                    type="GEN_V_SETPOINT",
                    gen_id=gen_id,
                    new_vm_pu=round(float(augmented_base.gen.at[gen_id, "vm_pu"]) - 0.02, 10),
                )
            ),
        ),
        (
            unavailable_shunt,
            _maneuver(
                ShuntStepAction(
                    type="SHUNT_STEP",
                    shunt_id=shunt_id,
                    new_step=1 - int(unavailable_shunt.shunt.at[shunt_id, "step"]),
                )
            ),
        ),
    ]

    for net, maneuver in cases:
        applicability = TopologyServer.get_action_applicability(net, maneuver.action)
        raised = _apply_raises_invalid_action(net, maneuver)

        assert applicability.applicable is False
        assert raised is True
        assert (not applicability.applicable) is raised


def test_b2_no_ops_are_rejected_by_soft_and_hard_gates(augmented_base):
    topology = TopologyServer.get_grid_topology(augmented_base)
    shunt = topology.shunts[0]
    trafo = _first_tappable_trafo_from_topology(topology)
    gen_id = _first_dispatchable_gen(augmented_base)
    gen_vm_pu = float(augmented_base.gen.at[gen_id, "vm_pu"])

    no_ops = [
        _maneuver(
            ShuntStepAction(
                type="SHUNT_STEP",
                shunt_id=shunt.shunt_id,
                new_step=shunt.step,
            )
        ),
        _maneuver(
            TapAdjustmentAction(
                type="TAP_ADJUSTMENT",
                trafo_id=trafo.trafo_id,
                new_tap_pos=trafo.tap_pos,
            )
        ),
        _maneuver(
            GenVoltageSetpointAction(
                type="GEN_V_SETPOINT",
                gen_id=gen_id,
                new_vm_pu=gen_vm_pu,
            )
        ),
    ]

    for maneuver in no_ops:
        applicability = TopologyServer.get_action_applicability(augmented_base, maneuver.action)
        handle = SandboxServer.create_sandbox(augmented_base)
        try:
            assert applicability.applicable is False
            assert applicability.reason is not None
            assert "no-op" in applicability.reason
            with pytest.raises(InvalidActionError):
                SandboxServer.apply_maneuver(handle, maneuver, saturated_gens=frozenset())
        finally:
            SandboxServer.discard_sandbox(handle)


def test_b2_valid_useful_action_is_soft_applicable_and_mutates_through_hard_gate(augmented_base):
    topology = TopologyServer.get_grid_topology(augmented_base)
    shunt = topology.shunts[0]
    target_step = 1 - shunt.step
    maneuver = _maneuver(
        ShuntStepAction(
            type="SHUNT_STEP",
            shunt_id=shunt.shunt_id,
            new_step=target_step,
        )
    )
    applicability = TopologyServer.get_action_applicability(augmented_base, maneuver.action)
    handle = SandboxServer.create_sandbox(augmented_base)
    try:
        raised = False
        try:
            SandboxServer.apply_maneuver(handle, maneuver, saturated_gens=frozenset())
        except InvalidActionError:
            raised = True
        resolved = SandboxServer.resolve_net(handle)

        assert applicability.applicable is True
        assert raised is False
        assert applicability.applicable is (not raised)
        assert int(resolved.shunt.at[shunt.shunt_id, "step"]) == target_step
        assert bool(resolved.shunt.at[shunt.shunt_id, "in_service"]) is shunt.in_service
    finally:
        SandboxServer.discard_sandbox(handle)


def test_d1_live_boundary_objects_use_the_shared_backend_schema_classes(diverging_scenario):
    pf_result = PowerFlowServer.run_ac_pf(diverging_scenario)
    assert pf_result.diagnostics is not None
    topology = TopologyServer.get_grid_topology(diverging_scenario)
    maneuver = _certified_max_reactive_maneuvers(diverging_scenario, tap_position=-2)[0]
    applicability = TopologyServer.get_action_applicability(diverging_scenario, maneuver.action)
    handle = SandboxServer.create_sandbox(diverging_scenario)
    try:
        assert isinstance(pf_result, schemas.PowerFlowResult)
        assert type(pf_result) is schemas.PowerFlowResult
        assert isinstance(pf_result.diagnostics, schemas.NRDiagnostics)
        assert type(pf_result.diagnostics) is schemas.NRDiagnostics
        assert isinstance(topology, schemas.TopologySummary)
        assert type(topology) is schemas.TopologySummary
        assert isinstance(handle, schemas.SandboxNet)
        assert type(handle) is schemas.SandboxNet
        assert isinstance(maneuver, schemas.Maneuver)
        assert type(maneuver) is schemas.Maneuver
        assert isinstance(applicability, schemas.ApplicabilityResult)
        assert type(applicability) is schemas.ApplicabilityResult
    finally:
        SandboxServer.discard_sandbox(handle)
