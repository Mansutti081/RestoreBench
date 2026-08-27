# ABOUTME: Verifies action and maneuver schema contracts.
# ABOUTME: Covers discriminated action union, bounds, extra rejection, and round trips.
import pytest
from pydantic import TypeAdapter, ValidationError

from restorebench.schemas.actions import (
    Action,
    GenVoltageSetpointAction,
    Maneuver,
    ManeuverSequence,
    ShuntStepAction,
    TapAdjustmentAction,
)


def round_trip(model):
    assert type(model).model_validate(model.model_dump()) == model
    assert type(model).model_validate_json(model.model_dump_json()) == model


def test_action_variants_round_trip_and_reject_extra_fields():
    actions = [
        GenVoltageSetpointAction(type="GEN_V_SETPOINT", gen_id=1, new_vm_pu=1.02),
        ShuntStepAction(type="SHUNT_STEP", shunt_id=3, new_step=0),
        TapAdjustmentAction(type="TAP_ADJUSTMENT", trafo_id=4, new_tap_pos=-1),
    ]

    for action in actions:
        round_trip(action)

    with pytest.raises(ValidationError):
        GenVoltageSetpointAction(type="GEN_V_SETPOINT", gen_id=1, new_vm_pu=1.0, extra=True)


def test_action_constraints_are_schema_level_only_where_specified():
    with pytest.raises(ValidationError):
        GenVoltageSetpointAction(type="GEN_V_SETPOINT", gen_id=1, new_vm_pu=1.20)

    with pytest.raises(ValidationError):
        TapAdjustmentAction(type="TAP_ADJUSTMENT", trafo_id=0, new_tap_pos=3)

    with pytest.raises(ValidationError):
        ShuntStepAction(type="SHUNT_STEP", shunt_id=0, new_step=2)


def test_discriminated_action_union_accepts_only_locked_variants():
    adapter = TypeAdapter(Action)

    parsed = adapter.validate_python({"type": "TAP_ADJUSTMENT", "trafo_id": 0, "new_tap_pos": 1})
    assert isinstance(parsed, TapAdjustmentAction)

    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "LINE_SWITCH", "line_id": 0, "target_state": False})
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "GEN_P_REDISPATCH", "gen_id": 0, "new_p_mw": 1.0})
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "SHUNT_SWITCHING", "shunt_id": 0, "target_state": True})


def test_maneuver_and_sequence_round_trip_without_schema_length_cap():
    maneuver = Maneuver(
        action={"type": "SHUNT_STEP", "shunt_id": 2, "new_step": 1},
        diagnosed_cause="REACTIVE_DEFICIT",
        rationale="Use shunt ${tool.shunt_state}.",
    )
    sequence = ManeuverSequence(maneuvers=[maneuver] * 25, reconstruction_summary=None)

    assert isinstance(maneuver.action, ShuntStepAction)
    assert len(sequence.maneuvers) == 25
    round_trip(maneuver)
    round_trip(sequence)
