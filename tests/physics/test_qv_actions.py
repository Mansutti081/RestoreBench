# ABOUTME: Verifies the canonical atomic Q-V action contract and shared action helpers.
# ABOUTME: Covers schema rejection, stable enumeration, applicability, and isolated application.
from __future__ import annotations

import pickle

import pytest
from pydantic import TypeAdapter, ValidationError

from restorebench.physics.actions import (
    apply_qv_action,
    enumerate_legal_qv_actions,
    get_qv_action_applicability,
)
from restorebench.schemas.actions import Action, GenVoltageSetpointAction, ShuntStepAction
from restorebench.tools.topology import get_action_applicability
from restorebench.corpus.augment import build_augmented_base


def _net_bytes(net) -> bytes:
    return pickle.dumps(net, protocol=pickle.HIGHEST_PROTOCOL)


def test_action_union_rejects_superseded_variants_and_round_trips_shunt_step() -> None:
    adapter = TypeAdapter(Action)

    for payload in (
        {"type": "GEN_P_REDISPATCH", "gen_id": 1, "new_p_mw": 100.0},
        {"type": "SHUNT_SWITCHING", "shunt_id": 1, "target_state": True},
    ):
        with pytest.raises(ValidationError):
            adapter.validate_python(payload)

    action = adapter.validate_python({"type": "SHUNT_STEP", "shunt_id": 1, "new_step": 0})

    assert isinstance(action, ShuntStepAction)
    assert ShuntStepAction.model_validate_json(action.model_dump_json()) == action


def test_enumerator_returns_only_atomic_actions_in_stable_order() -> None:
    net = build_augmented_base()
    gen_ids = [int(gen_id) for gen_id in net.gen.index[:2]]
    shunt_id = int(net.shunt.index[0])
    trafo_id = int(net.trafo.index[net.trafo["tap_pos"].notna()][0])

    net.gen.loc[:, "in_service"] = False
    net.gen.loc[gen_ids, "in_service"] = True
    net.gen.at[gen_ids[0], "vm_pu"] = 1.00
    net.gen.at[gen_ids[1], "vm_pu"] = 0.95

    net.shunt.loc[:, "in_service"] = False
    net.shunt.at[shunt_id, "in_service"] = True
    net.shunt.at[shunt_id, "step"] = 0
    net.shunt.at[shunt_id, "max_step"] = 1

    net.trafo.loc[:, "in_service"] = False
    net.trafo.at[trafo_id, "in_service"] = True
    net.trafo.at[trafo_id, "tap_pos"] = 0
    net.trafo.at[trafo_id, "tap_min"] = -1
    net.trafo.at[trafo_id, "tap_max"] = 1

    actions = enumerate_legal_qv_actions(
        net,
        q_context={gen_ids[0]: "Q_LIMITED_UPPER", gen_ids[1]: "PV_CONTROLLABLE"},
    )

    assert [action.model_dump(mode="json") for action in actions] == [
        {"type": "GEN_V_SETPOINT", "gen_id": gen_ids[0], "new_vm_pu": 0.99},
        {"type": "GEN_V_SETPOINT", "gen_id": gen_ids[1], "new_vm_pu": 0.96},
        {"type": "SHUNT_STEP", "shunt_id": shunt_id, "new_step": 1},
        {"type": "TAP_ADJUSTMENT", "trafo_id": trafo_id, "new_tap_pos": -1},
        {"type": "TAP_ADJUSTMENT", "trafo_id": trafo_id, "new_tap_pos": 1},
    ]


def test_applicability_rejects_non_atomic_targets() -> None:
    net = build_augmented_base()
    gen_id = int(net.gen.index[0])
    trafo_id = int(net.trafo.index[net.trafo["tap_pos"].notna()][0])
    net.gen.at[gen_id, "vm_pu"] = 1.00
    net.trafo.at[trafo_id, "tap_pos"] = 0

    non_atomic_gen = TypeAdapter(Action).validate_python(
        {"type": "GEN_V_SETPOINT", "gen_id": gen_id, "new_vm_pu": 1.02}
    )
    non_atomic_tap = TypeAdapter(Action).validate_python(
        {"type": "TAP_ADJUSTMENT", "trafo_id": trafo_id, "new_tap_pos": 2}
    )

    assert get_action_applicability(net, non_atomic_gen).applicable is False
    assert get_action_applicability(net, non_atomic_tap).applicable is False


def test_generator_outside_action_band_has_no_clipped_atomic_move() -> None:
    net = build_augmented_base()
    gen_id = int(net.gen.index[0])
    net.gen.at[gen_id, "vm_pu"] = 1.07
    action = GenVoltageSetpointAction(
        type="GEN_V_SETPOINT",
        gen_id=gen_id,
        new_vm_pu=1.05,
    )

    applicability = get_qv_action_applicability(net, action)
    enumerated = enumerate_legal_qv_actions(net)

    assert applicability.applicable is False
    assert "outside" in (applicability.reason or "")
    assert all(candidate.type != "GEN_V_SETPOINT" or candidate.gen_id != gen_id for candidate in enumerated)


def test_enumerator_skips_tap_targets_outside_the_action_schema_domain() -> None:
    # A net may declare wider tap bounds than the action schema allows. Enumeration must
    # skip those targets, because constructing the action would raise instead of yielding
    # the legal subset.
    net = build_augmented_base()
    trafo_id = int(net.trafo.index[net.trafo["tap_pos"].notna()][0])
    net.trafo.loc[:, "in_service"] = False
    net.trafo.at[trafo_id, "in_service"] = True
    net.trafo.at[trafo_id, "tap_min"] = -9
    net.trafo.at[trafo_id, "tap_max"] = 9
    net.trafo.at[trafo_id, "tap_pos"] = 8

    enumerated = enumerate_legal_qv_actions(net)

    assert all(candidate.type != "TAP_ADJUSTMENT" for candidate in enumerated)


def test_enumerator_keeps_tap_targets_that_remain_inside_the_schema_domain() -> None:
    net = build_augmented_base()
    trafo_id = int(net.trafo.index[net.trafo["tap_pos"].notna()][0])
    net.trafo.loc[:, "in_service"] = False
    net.trafo.at[trafo_id, "in_service"] = True
    net.trafo.at[trafo_id, "tap_min"] = -9
    net.trafo.at[trafo_id, "tap_max"] = 9
    net.trafo.at[trafo_id, "tap_pos"] = 2

    taps = [candidate for candidate in enumerate_legal_qv_actions(net) if candidate.type == "TAP_ADJUSTMENT"]

    # tap_pos 2 -> +1 leaves the schema domain, -1 stays inside it.
    assert [candidate.new_tap_pos for candidate in taps] == [1]


def test_apply_shunt_step_preserves_structural_availability_and_caller() -> None:
    net = build_augmented_base()
    shunt_id = int(net.shunt.index[0])
    net.shunt.at[shunt_id, "in_service"] = True
    net.shunt.at[shunt_id, "step"] = 0
    before = _net_bytes(net)

    action = ShuntStepAction(type="SHUNT_STEP", shunt_id=shunt_id, new_step=1)
    changed = apply_qv_action(net, action, q_context={})

    assert changed is not net
    assert int(changed.shunt.at[shunt_id, "step"]) == 1
    assert bool(changed.shunt.at[shunt_id, "in_service"]) is True
    assert _net_bytes(net) == before
