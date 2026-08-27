# ABOUTME: Verifies iteration feedback and sandbox-handle schemas.
# ABOUTME: Covers round trips, extra rejection, and opaque UUID sandbox handles.
from uuid import uuid4

import pytest
from pydantic import ValidationError

from restorebench.schemas.feedback import FailureFeedback, SandboxNet
from restorebench.schemas.power_flow import NRDiagnostics


def round_trip(model):
    assert type(model).model_validate(model.model_dump()) == model
    assert type(model).model_validate_json(model.model_dump_json()) == model


def diagnostics() -> NRDiagnostics:
    return NRDiagnostics(
        iterations_attempted=30,
        worst_bus=7,
        lowest_vm_pu=0.8,
        lowest_vm_bus=7,
        gens_at_q_limit=[],
        max_mismatch_mw=None,
        max_mismatch_mvar=None,
        overstress=None,
        error_message="failed",
        diagnostics_source="local_nose",
    )


def test_feedback_models_round_trip():
    feedback = FailureFeedback(
        iteration=1,
        kind="STILL_DIVERGED",
        diagnostics=diagnostics(),
        detail=None,
        maneuver={
            "action": {"type": "TAP_ADJUSTMENT", "trafo_id": 0, "new_tap_pos": 1},
            "diagnosed_cause": "BAD_SETPOINTS",
            "rationale": "Adjust tap.",
        },
    )
    sandbox = SandboxNet(sandbox_id=uuid4(), scenario_request_id=uuid4())

    round_trip(feedback)
    round_trip(sandbox)


def test_solved_infeasible_feedback_round_trips_without_diagnostics():
    feedback = FailureFeedback(
        iteration=1,
        kind="SOLVED_INFEASIBLE",
        diagnostics=None,
        detail="external-grid Q exceeds its declared bound",
        maneuver=None,
    )

    round_trip(feedback)


def test_feedback_rejects_extra_fields():
    with pytest.raises(ValidationError):
        SandboxNet(sandbox_id=uuid4(), scenario_request_id=uuid4(), net="not allowed")
