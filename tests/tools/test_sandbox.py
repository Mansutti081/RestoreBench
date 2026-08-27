# ABOUTME: Verifies SandboxServer copy isolation, action bounds, and handle lifecycle.
# ABOUTME: Uses the augmented IEEE-118 base to exercise real pandapower element tables.
from __future__ import annotations

import pickle
from uuid import uuid4

import pytest

from restorebench.schemas import (
    GenVoltageSetpointAction,
    InvalidActionError,
    Maneuver,
    SandboxNet,
    ShuntStepAction,
    TapAdjustmentAction,
    ToolFailureError,
)
from restorebench.tools.sandbox import (
    apply_maneuver,
    create_sandbox,
    discard_sandbox,
    promote_sandbox,
    resolve_net,
)
from restorebench.corpus.augment import build_augmented_base


def _net_bytes(net) -> bytes:
    return pickle.dumps(net, protocol=pickle.HIGHEST_PROTOCOL)


def _maneuver(action) -> Maneuver:
    return Maneuver(action=action, diagnosed_cause=None, rationale="test maneuver")


@pytest.fixture
def net():
    return build_augmented_base()


def _first_tappable_trafo(net) -> int:
    mask = net.trafo["in_service"].astype(bool) & net.trafo["tap_pos"].notna()
    return int(net.trafo.index[mask][0])


def _first_non_tappable_trafo(net) -> int:
    mask = net.trafo["in_service"].astype(bool) & net.trafo["tap_pos"].isna()
    return int(net.trafo.index[mask][0])


def test_create_sandbox_returns_handle_and_preserves_caller_net_byte_identity(net):
    before = _net_bytes(net)

    sandbox = create_sandbox(net)
    try:
        sandbox_net = resolve_net(sandbox)

        assert isinstance(sandbox, SandboxNet)
        assert sandbox.scenario_request_id is not None
        assert sandbox_net is not net
        assert _net_bytes(net) == before
    finally:
        discard_sandbox(sandbox)


def test_resolve_net_returns_direct_net_without_wrapping(net):
    assert resolve_net(net) is net


def test_gen_voltage_setpoint_updates_sandbox_only(net):
    gen_id = int(net.gen.index[net.gen["in_service"].astype(bool)][0])
    current_vm_pu = float(net.gen.at[gen_id, "vm_pu"])
    new_vm_pu = round(current_vm_pu + 0.01, 10) if current_vm_pu < 1.05 else round(current_vm_pu - 0.01, 10)
    before = _net_bytes(net)
    sandbox = create_sandbox(net)

    try:
        returned = apply_maneuver(
            sandbox,
            _maneuver(
                GenVoltageSetpointAction(
                    type="GEN_V_SETPOINT",
                    gen_id=gen_id,
                    new_vm_pu=new_vm_pu,
                )
            ),
            saturated_gens=frozenset(),
        )

        assert returned == sandbox
        assert float(resolve_net(sandbox).gen.at[gen_id, "vm_pu"]) == pytest.approx(new_vm_pu)
        assert _net_bytes(net) == before
    finally:
        discard_sandbox(sandbox)


def test_gen_voltage_setpoint_rejects_missing_and_out_of_service_gens(net):
    gen_id = int(net.gen.index[net.gen["in_service"].astype(bool)][0])
    net.gen.at[gen_id, "in_service"] = False
    before = _net_bytes(net)
    sandbox = create_sandbox(net)

    try:
        with pytest.raises(InvalidActionError):
            apply_maneuver(
                sandbox,
                _maneuver(
                    GenVoltageSetpointAction(
                        type="GEN_V_SETPOINT",
                        gen_id=gen_id,
                        new_vm_pu=round(float(net.gen.at[gen_id, "vm_pu"]) + 0.01, 10),
                    )
                ),
                saturated_gens=frozenset(),
            )

        with pytest.raises(InvalidActionError):
            apply_maneuver(
                sandbox,
                _maneuver(
                    GenVoltageSetpointAction(
                        type="GEN_V_SETPOINT",
                        gen_id=int(net.gen.index.max()) + 1000,
                        new_vm_pu=1.0,
                    )
                ),
                saturated_gens=frozenset(),
            )

        assert _net_bytes(net) == before
    finally:
        discard_sandbox(sandbox)


def test_gen_voltage_setpoint_rejects_raise_for_q_saturated_gen_and_allows_lower(net):
    gen_id = int(net.gen.index[net.gen["in_service"].astype(bool)][0])
    net.gen.at[gen_id, "vm_pu"] = 1.0
    sandbox = create_sandbox(net)

    try:
        with pytest.raises(InvalidActionError, match="Q-saturated"):
            apply_maneuver(
                sandbox,
                _maneuver(
                    GenVoltageSetpointAction(
                        type="GEN_V_SETPOINT",
                        gen_id=gen_id,
                        new_vm_pu=1.01,
                    )
                ),
                saturated_gens=frozenset({gen_id}),
            )

        assert float(resolve_net(sandbox).gen.at[gen_id, "vm_pu"]) == pytest.approx(1.0)

        apply_maneuver(
            sandbox,
            _maneuver(
                GenVoltageSetpointAction(
                    type="GEN_V_SETPOINT",
                    gen_id=gen_id,
                    new_vm_pu=0.99,
                )
            ),
            saturated_gens=frozenset({gen_id}),
        )

        assert float(resolve_net(sandbox).gen.at[gen_id, "vm_pu"]) == pytest.approx(0.99)
    finally:
        discard_sandbox(sandbox)


def test_shunt_step_updates_step_only_and_rejects_unknown_shunt(net):
    shunt_id = int(net.shunt.index[0])
    net.shunt.at[shunt_id, "step"] = 0
    new_step = 1
    immutable_device_data = net.shunt.loc[shunt_id, ["q_mvar", "max_step", "in_service"]].copy()
    before = _net_bytes(net)
    sandbox = create_sandbox(net)

    try:
        apply_maneuver(
            sandbox,
            _maneuver(
                ShuntStepAction(
                    type="SHUNT_STEP",
                    shunt_id=shunt_id,
                    new_step=new_step,
                )
            ),
            saturated_gens=frozenset(),
        )

        sandbox_shunt = resolve_net(sandbox).shunt.loc[shunt_id]
        assert int(sandbox_shunt["step"]) == new_step
        assert bool(sandbox_shunt["in_service"]) is bool(immutable_device_data["in_service"])
        assert sandbox_shunt["q_mvar"] == immutable_device_data["q_mvar"]
        assert sandbox_shunt["max_step"] == immutable_device_data["max_step"]

        with pytest.raises(InvalidActionError):
            apply_maneuver(
                sandbox,
                _maneuver(
                    ShuntStepAction(
                        type="SHUNT_STEP",
                        shunt_id=int(net.shunt.index.max()) + 1000,
                        new_step=1,
                    )
                ),
                saturated_gens=frozenset(),
            )

        assert _net_bytes(net) == before
    finally:
        discard_sandbox(sandbox)


def test_tap_adjustment_applies_for_tappable_trafo_and_rejects_invalid_targets(net):
    trafo_id = _first_tappable_trafo(net)
    tap_max = int(net.trafo.at[trafo_id, "tap_max"])
    current_tap_pos = int(net.trafo.at[trafo_id, "tap_pos"])
    new_tap_pos = current_tap_pos + 1 if current_tap_pos < tap_max else current_tap_pos - 1
    before = _net_bytes(net)
    sandbox = create_sandbox(net)

    try:
        apply_maneuver(
            sandbox,
            _maneuver(
                TapAdjustmentAction(
                    type="TAP_ADJUSTMENT",
                    trafo_id=trafo_id,
                    new_tap_pos=new_tap_pos,
                )
            ),
            saturated_gens=frozenset(),
        )

        assert int(resolve_net(sandbox).trafo.at[trafo_id, "tap_pos"]) == new_tap_pos

        net_with_narrow_bound = build_augmented_base()
        narrow_id = _first_tappable_trafo(net_with_narrow_bound)
        net_with_narrow_bound.trafo.at[narrow_id, "tap_max"] = 1
        narrow_sandbox = create_sandbox(net_with_narrow_bound)
        try:
            with pytest.raises(InvalidActionError):
                apply_maneuver(
                    narrow_sandbox,
                    _maneuver(
                        TapAdjustmentAction(
                            type="TAP_ADJUSTMENT",
                            trafo_id=narrow_id,
                            new_tap_pos=2,
                        )
                    ),
                    saturated_gens=frozenset(),
                )
        finally:
            discard_sandbox(narrow_sandbox)

        assert _net_bytes(net) == before
    finally:
        discard_sandbox(sandbox)


def test_tap_adjustment_rejects_missing_out_of_service_and_non_tappable_trafos(net):
    tappable_id = _first_tappable_trafo(net)
    non_tappable_id = _first_non_tappable_trafo(net)
    net.trafo.at[tappable_id, "in_service"] = False
    before = _net_bytes(net)
    sandbox = create_sandbox(net)

    try:
        for trafo_id in (
            tappable_id,
            non_tappable_id,
            int(net.trafo.index.max()) + 1000,
        ):
            with pytest.raises(InvalidActionError):
                apply_maneuver(
                    sandbox,
                    _maneuver(
                        TapAdjustmentAction(
                            type="TAP_ADJUSTMENT",
                            trafo_id=trafo_id,
                            new_tap_pos=0,
                        )
                    ),
                    saturated_gens=frozenset(),
                )

        assert _net_bytes(net) == before
    finally:
        discard_sandbox(sandbox)


def test_two_sandboxes_from_same_net_are_independent(net):
    gen_id = int(net.gen.index[net.gen["in_service"].astype(bool)][0])
    current_vm_pu = float(net.gen.at[gen_id, "vm_pu"])
    new_vm_pu = round(current_vm_pu + 0.01, 10) if current_vm_pu < 1.05 else round(current_vm_pu - 0.01, 10)
    first = create_sandbox(net)
    second = create_sandbox(net)

    try:
        apply_maneuver(
            first,
            _maneuver(
                GenVoltageSetpointAction(
                    type="GEN_V_SETPOINT",
                    gen_id=gen_id,
                    new_vm_pu=new_vm_pu,
                )
            ),
            saturated_gens=frozenset(),
        )

        assert float(resolve_net(first).gen.at[gen_id, "vm_pu"]) == pytest.approx(new_vm_pu)
        assert float(resolve_net(second).gen.at[gen_id, "vm_pu"]) == pytest.approx(float(net.gen.at[gen_id, "vm_pu"]))
    finally:
        discard_sandbox(first)
        discard_sandbox(second)


def test_promote_discard_and_resolve_lifecycle(net):
    sandbox = create_sandbox(net, scenario_request_id=uuid4())

    promote_sandbox(sandbox)
    discard_sandbox(sandbox)

    with pytest.raises(ToolFailureError):
        resolve_net(sandbox)


def test_unknown_handles_raise_tool_failure_for_apply_promote_and_resolve():
    missing = SandboxNet(sandbox_id=uuid4(), scenario_request_id=uuid4())
    maneuver = _maneuver(
        ShuntStepAction(
            type="SHUNT_STEP",
            shunt_id=0,
            new_step=0,
        )
    )

    with pytest.raises(ToolFailureError):
        resolve_net(missing)

    with pytest.raises(ToolFailureError):
        apply_maneuver(missing, maneuver, saturated_gens=frozenset())

    with pytest.raises(ToolFailureError):
        promote_sandbox(missing)
