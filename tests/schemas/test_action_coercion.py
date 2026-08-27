# ABOUTME: Guards the boundary coercion for actions Bedrock returns as JSON strings.
# ABOUTME: Bedrock models serialize a oneOf object property as a string; the decision inside it is still valid.
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from restorebench.agents.tool_loop import ActionApplicabilityInput
from restorebench.schemas.actions import GenVoltageSetpointAction, Maneuver


ACTION_PAYLOAD = {"type": "GEN_V_SETPOINT", "gen_id": 11, "new_vm_pu": 1.05}


def test_maneuver_accepts_an_action_bedrock_returned_as_a_json_string() -> None:
    maneuver = Maneuver.model_validate(
        {
            "action": json.dumps(ACTION_PAYLOAD),
            "diagnosed_cause": "REACTIVE_DEFICIT",
            "rationale": "raise reactive support at the collapsed bus",
        }
    )

    assert maneuver.action == GenVoltageSetpointAction.model_validate(ACTION_PAYLOAD)


def test_action_applicability_input_accepts_an_action_returned_as_a_json_string() -> None:
    parsed = ActionApplicabilityInput.model_validate({"action": json.dumps(ACTION_PAYLOAD)})

    assert parsed.action == GenVoltageSetpointAction.model_validate(ACTION_PAYLOAD)


def test_action_applicability_input_unwraps_a_double_wrapped_action() -> None:
    # Bedrock models echo the wrapper key back inside itself: {"action": {"action": {...}}}.
    parsed = ActionApplicabilityInput.model_validate({"action": {"action": ACTION_PAYLOAD}})

    assert parsed.action == GenVoltageSetpointAction.model_validate(ACTION_PAYLOAD)


def test_a_json_object_action_is_still_accepted_unchanged() -> None:
    parsed = ActionApplicabilityInput.model_validate({"action": ACTION_PAYLOAD})

    assert parsed.action == GenVoltageSetpointAction.model_validate(ACTION_PAYLOAD)


def test_an_action_string_that_is_not_json_is_a_validation_error() -> None:
    with pytest.raises(ValidationError):
        ActionApplicabilityInput.model_validate({"action": "raise generator 11 to 1.05"})


def test_an_out_of_bounds_action_string_is_still_rejected() -> None:
    # Coercion decodes the transport shape; it must never relax the action bounds.
    out_of_bounds = json.dumps({"type": "GEN_V_SETPOINT", "gen_id": 11, "new_vm_pu": 1.40})

    with pytest.raises(ValidationError):
        ActionApplicabilityInput.model_validate({"action": out_of_bounds})
