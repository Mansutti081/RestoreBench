# ABOUTME: Covers the coercion that repairs JSON-encoded nested objects in tool payloads.
# ABOUTME: A model that stringifies a nested field has answered correctly in the wrong encoding.
import pytest
from pydantic import BaseModel, ValidationError

from restorebench.agents.structured_output import coerce_nested_json
from restorebench.schemas.actions import Maneuver, ManeuverSequence
from restorebench.schemas.multi_agent import AnalystAssessment

# The exact shape DeepSeek V3.2 returned on every one of ten iterations of the E3 gate:
# semantically correct content, with the nested object serialised as a JSON string.
DEEPSEEK_PAYLOAD = {
    "diagnosed_cause": "REACTIVE_DEFICIT",
    "proposed_maneuver": (
        '{"action": {"type": "SHUNT_STEP", "shunt_id": 0, "new_step": 0}, '
        '"diagnosed_cause": "REACTIVE_DEFICIT", "rationale": "Remove the reactor."}'
    ),
    "rationale": "The weak bus sits inside the reactor's pocket.",
}


def test_a_stringified_nested_object_validates_after_coercion() -> None:
    with pytest.raises(ValidationError):
        AnalystAssessment.model_validate(DEEPSEEK_PAYLOAD)

    assessment = AnalystAssessment.model_validate(
        coerce_nested_json(DEEPSEEK_PAYLOAD, AnalystAssessment)
    )

    assert assessment.proposed_maneuver.action.type == "SHUNT_STEP"
    assert assessment.proposed_maneuver.action.shunt_id == 0
    assert assessment.rationale.startswith("The weak bus")


def test_a_well_formed_payload_is_returned_unchanged() -> None:
    payload = {
        "diagnosed_cause": "REACTIVE_DEFICIT",
        "proposed_maneuver": {
            "action": {"type": "SHUNT_STEP", "shunt_id": 0, "new_step": 0},
            "diagnosed_cause": "REACTIVE_DEFICIT",
            "rationale": "Remove the reactor.",
        },
        "rationale": "unchanged",
    }

    assert coerce_nested_json(payload, AnalystAssessment) == payload


def test_a_string_field_that_looks_like_json_is_left_alone() -> None:
    """Only fields whose annotation is a model may be decoded. `rationale` is a string field,
    and parsing it would silently replace the model's own words with a dict."""
    payload = {
        "diagnosed_cause": "REACTIVE_DEFICIT",
        "proposed_maneuver": {
            "action": {"type": "SHUNT_STEP", "shunt_id": 0, "new_step": 0},
            "diagnosed_cause": "REACTIVE_DEFICIT",
            "rationale": "Remove the reactor.",
        },
        "rationale": '{"not": "a maneuver"}',
    }

    assert coerce_nested_json(payload, AnalystAssessment)["rationale"] == '{"not": "a maneuver"}'


def test_a_stringified_list_of_models_is_decoded() -> None:
    payload = {
        "maneuvers": (
            '[{"action": {"type": "SHUNT_STEP", "shunt_id": 0, "new_step": 0}, '
            '"diagnosed_cause": "REACTIVE_DEFICIT", "rationale": "first"}]'
        ),
        "reconstruction_summary": None,
    }

    sequence = ManeuverSequence.model_validate(coerce_nested_json(payload, ManeuverSequence))

    assert len(sequence.maneuvers) == 1
    assert sequence.maneuvers[0].rationale == "first"


def test_a_string_that_is_not_json_is_left_for_the_validator_to_reject() -> None:
    """Coercion repairs an encoding, never a meaning: unparseable input must still fail loudly
    rather than be dropped or guessed at."""
    payload = {"action": "not json at all", "diagnosed_cause": None, "rationale": "x"}

    coerced = coerce_nested_json(payload, Maneuver)

    assert coerced["action"] == "not json at all"
    with pytest.raises(ValidationError):
        Maneuver.model_validate(coerced)


def test_a_payload_that_is_not_a_mapping_is_passed_through() -> None:
    assert coerce_nested_json("nonsense", Maneuver) == "nonsense"
    assert coerce_nested_json(None, Maneuver) is None


def test_coercion_does_not_recurse_into_already_decoded_objects() -> None:
    """A nested dict is handed to pydantic as-is. Walking deeper would let a string buried in a
    rationale be reinterpreted, which is the failure mode this function must not have."""

    class Outer(BaseModel):
        inner: Maneuver

    payload = {
        "inner": {
            "action": {"type": "SHUNT_STEP", "shunt_id": 0, "new_step": 0},
            "diagnosed_cause": "REACTIVE_DEFICIT",
            "rationale": '{"looks": "like json"}',
        }
    }

    coerced = coerce_nested_json(payload, Outer)

    assert coerced["inner"]["rationale"] == '{"looks": "like json"}'
