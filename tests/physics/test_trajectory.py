# ABOUTME: Verifies immutable curation and snapshot-anchored diagnostic trajectory reconstruction.
# ABOUTME: Keeps private curation inputs structurally absent from the public diagnostic request.
from __future__ import annotations

import copy
from typing import Any

import pandapower as pp
import pandas as pd
import pytest
from pydantic import ValidationError

from restorebench.physics.actions import apply_qv_action
from restorebench.physics.trajectory import build_curation_state, build_diagnostic_state
from restorebench.schemas.actions import GenVoltageSetpointAction, ShuntStepAction, TapAdjustmentAction
from restorebench.schemas.physics import (
    ActiveBalancePolicy,
    CurationLoadWeight,
    CurationTrajectoryRequest,
    DiagnosticTrajectoryRequest,
    GeneratorParticipation,
)


def _trajectory_net() -> Any:
    net = pp.create_empty_network(sn_mva=100.0)
    slack = pp.create_bus(net, vn_kv=110.0)
    load_bus = pp.create_bus(net, vn_kv=110.0)
    pp.create_ext_grid(
        net,
        bus=slack,
        vm_pu=1.0,
        min_p_mw=-100.0,
        max_p_mw=100.0,
        min_q_mvar=-100.0,
        max_q_mvar=100.0,
    )
    pp.create_line_from_parameters(
        net,
        from_bus=slack,
        to_bus=load_bus,
        length_km=1.0,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.2,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
    )
    pp.create_gen(
        net,
        bus=load_bus,
        p_mw=30.0,
        vm_pu=1.0,
        min_p_mw=10.0,
        max_p_mw=80.0,
        min_q_mvar=-40.0,
        max_q_mvar=40.0,
        index=2,
    )
    pp.create_load(net, bus=load_bus, p_mw=20.0, q_mvar=8.0, index=3)
    pp.create_load(net, bus=load_bus, p_mw=10.0, q_mvar=-2.0, index=8)
    pp.create_shunt(net, bus=load_bus, q_mvar=-5.0, p_mw=0.0, step=0, max_step=1, index=4)
    pp.create_transformer_from_parameters(
        net,
        hv_bus=slack,
        lv_bus=load_bus,
        sn_mva=100.0,
        vn_hv_kv=110.0,
        vn_lv_kv=110.0,
        vk_percent=10.0,
        vkr_percent=0.5,
        pfe_kw=0.0,
        i0_percent=0.0,
        tap_side="hv",
        tap_neutral=0,
        tap_min=-2,
        tap_max=2,
        tap_step_percent=1.0,
        tap_pos=0,
        index=6,
    )
    return net


def _active_policy() -> ActiveBalancePolicy:
    return ActiveBalancePolicy(
        participation=(GeneratorParticipation(gen_id=2, factor=1.0),),
    )


def _weights() -> tuple[CurationLoadWeight, ...]:
    return (
        CurationLoadWeight(load_id=3, weight=1.0),
        CurationLoadWeight(load_id=8, weight=0.25),
    )


def test_curation_stress_zero_reconstructs_profile_without_mutating_it() -> None:
    profile = _trajectory_net()
    before = copy.deepcopy(profile)

    state = build_curation_state(
        profile,
        stress=0.0,
        ordered_load_weights=_weights(),
        active_policy=_active_policy(),
    )

    pd.testing.assert_frame_equal(state.net.load, before.load, check_exact=True)
    pd.testing.assert_frame_equal(state.net.gen, before.gen, check_exact=True)
    pd.testing.assert_frame_equal(profile.load, before.load, check_exact=True)
    pd.testing.assert_frame_equal(profile.gen, before.gen, check_exact=True)
    assert state.requested_load_delta_mw == 0.0


def test_diagnostic_lambda_one_preserves_promoted_qv_actions_exactly() -> None:
    snapshot = _trajectory_net()
    snapshot = apply_qv_action(
        snapshot,
        GenVoltageSetpointAction(type="GEN_V_SETPOINT", gen_id=2, new_vm_pu=1.01),
    )
    snapshot = apply_qv_action(
        snapshot,
        ShuntStepAction(type="SHUNT_STEP", shunt_id=4, new_step=1),
    )
    snapshot = apply_qv_action(
        snapshot,
        TapAdjustmentAction(type="TAP_ADJUSTMENT", trafo_id=6, new_tap_pos=1),
    )

    state = build_diagnostic_state(
        snapshot,
        lambda_value=1.0,
        active_policy=_active_policy(),
    )

    pd.testing.assert_frame_equal(state.net.load, snapshot.load, check_exact=True)
    pd.testing.assert_frame_equal(state.net.gen, snapshot.gen, check_exact=True)
    pd.testing.assert_frame_equal(state.net.shunt, snapshot.shunt, check_exact=True)
    pd.testing.assert_frame_equal(state.net.trafo, snapshot.trafo, check_exact=True)


@pytest.mark.parametrize("coordinate", [0.4, 1.6])
def test_trajectory_scaling_preserves_each_load_power_factor(coordinate: float) -> None:
    reference = _trajectory_net()
    curation = build_curation_state(
        reference,
        stress=coordinate,
        ordered_load_weights=_weights(),
        active_policy=_active_policy(),
    )
    diagnostic = build_diagnostic_state(
        reference,
        lambda_value=coordinate,
        active_policy=_active_policy(),
    )

    for state in (curation, diagnostic):
        for load_id in reference.load.index:
            base_p = float(reference.load.at[load_id, "p_mw"])
            base_q = float(reference.load.at[load_id, "q_mvar"])
            target_p = float(state.net.load.at[load_id, "p_mw"])
            target_q = float(state.net.load.at[load_id, "q_mvar"])
            assert target_p * base_q == pytest.approx(target_q * base_p, abs=1e-10)


def test_trajectory_calls_are_order_independent_and_reference_anchored() -> None:
    reference = _trajectory_net()

    high_first = build_curation_state(
        reference,
        stress=1.5,
        ordered_load_weights=_weights(),
        active_policy=_active_policy(),
    )
    low_after = build_curation_state(
        reference,
        stress=0.2,
        ordered_load_weights=_weights(),
        active_policy=_active_policy(),
    )
    low_direct = build_curation_state(
        copy.deepcopy(reference),
        stress=0.2,
        ordered_load_weights=_weights(),
        active_policy=_active_policy(),
    )

    assert not high_first.net.load[["p_mw", "q_mvar"]].equals(low_after.net.load[["p_mw", "q_mvar"]])
    pd.testing.assert_frame_equal(low_after.net.load, low_direct.net.load, check_exact=True)
    pd.testing.assert_frame_equal(low_after.net.gen, low_direct.net.gen, check_exact=True)
    assert list(reference.load["p_mw"]) == [20.0, 10.0]


def test_trajectory_requests_reject_mismatched_ids_and_illegal_coordinates() -> None:
    reference = _trajectory_net()
    policy = _active_policy()

    with pytest.raises(ValueError, match="load IDs"):
        build_curation_state(
            reference,
            stress=0.1,
            ordered_load_weights=tuple(reversed(_weights())),
            active_policy=policy,
        )
    with pytest.raises(ValidationError):
        build_curation_state(
            reference,
            stress=-0.1,
            ordered_load_weights=_weights(),
            active_policy=policy,
        )
    with pytest.raises(ValidationError):
        build_diagnostic_state(reference, lambda_value=0.0, active_policy=policy)
    with pytest.raises(ValidationError):
        build_curation_state(
            reference,
            stress=float("inf"),
            ordered_load_weights=_weights(),
            active_policy=policy,
        )
    with pytest.raises(ValidationError):
        build_diagnostic_state(reference, lambda_value=float("inf"), active_policy=policy)


def test_diagnostic_request_schema_contains_no_private_curation_fields() -> None:
    assert set(CurationTrajectoryRequest.model_fields) == {"stress", "ordered_load_weights"}
    assert set(DiagnosticTrajectoryRequest.model_fields) == {"lambda_value"}
    diagnostic_schema = str(DiagnosticTrajectoryRequest.model_json_schema()).lower()
    for private_name in ("anchor", "weight", "stress", "boundary", "witness"):
        assert private_name not in diagnostic_schema
