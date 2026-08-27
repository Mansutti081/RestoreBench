# ABOUTME: Verifies response, trace, citation, and resolution schemas.
# ABOUTME: Covers timezone-aware UTC datetimes, status/converged consistency, and round trips.
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from restorebench.schemas.response import (
    RESULT_SCHEMA_VERSION,
    Citation,
    ExecutionTrace,
    ReasoningEntry,
    ResolutionResponse,
    TraceEvent,
)


def round_trip(model):
    assert type(model).model_validate(model.model_dump()) == model
    assert type(model).model_validate_json(model.model_dump_json()) == model


def trace_event(**overrides) -> TraceEvent:
    data = {
        "timestamp": datetime(2026, 7, 5, 12, 0, tzinfo=timezone.utc),
        "phase": "solve",
        "event_name": "run_ac_pf",
        "duration_ms": 12.0,
        "payload": {"iterations": 7},
    }
    data.update(overrides)
    return TraceEvent(**data)


def execution_trace() -> ExecutionTrace:
    return ExecutionTrace(
        request_id=uuid4(),
        events=[trace_event()],
        n_llm_calls=1,
        total_llm_tokens_in=10,
        total_llm_tokens_out=5,
        n_tool_calls=2,
        n_power_flows=3,
    )


def resolution_response(**overrides) -> ResolutionResponse:
    data = {
        "request_id": uuid4(),
        "scenario_id": "S0001",
        "configuration": 2,
        "llm_assignment": {"single_agent": "qwen3:4b"},
        "repetition_index": 3,
        "dataset_version": "reactive-deficit-v1",
        "solver_version": "locked-nr-q-limited-v1",
        "action_policy_version": "qv-atomic-v1",
        "ranking_policy_version": "snapshot-anchored-retreat-v1",
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "status": "SUCCESS",
        "maneuvers": [
            {
                "action": {"type": "GEN_V_SETPOINT", "gen_id": 1, "new_vm_pu": 1.02},
                "diagnosed_cause": "REACTIVE_DEFICIT",
                "rationale": "Raise voltage.",
            }
        ],
        "n_maneuvers": 1,
        "converged": True,
        "quality": {
            "clean": True,
            "n_buses_out_of_band": 0,
            "worst_vm_pu": 0.99,
            "worst_vm_bus": 4,
            "symptoms": [],
        },
        "final_warnings": [],
        "diagnosis_rationale": None,
        "citations": [],
        "failure_feedback": [],
        "trace": execution_trace(),
        "total_runtime_seconds": 5.0,
        "started_at": datetime(2026, 7, 5, 12, 0, tzinfo=timezone.utc),
        "completed_at": datetime(2026, 7, 5, 12, 1, tzinfo=timezone.utc),
    }
    data.update(overrides)
    return ResolutionResponse(**data)


def test_response_models_round_trip():
    models = [
        trace_event(),
        ReasoningEntry(iteration=2, role="analyst", text="Voltage collapse near bus 44."),
        execution_trace(),
        Citation(
            marker_id="${diag.lowest_vm_pu}",
            solver_field_path="iterations[3].diagnostics.lowest_vm_pu",
            resolved_value=0.91,
            formatting=".3f",
        ),
        resolution_response(),
    ]

    for model in models:
        round_trip(model)


def test_datetimes_must_be_timezone_aware_utc():
    with pytest.raises(ValidationError):
        trace_event(timestamp=datetime(2026, 7, 5, 12, 0))

    shifted = trace_event(timestamp=datetime(2026, 7, 5, 14, 0, tzinfo=timezone.utc))
    assert shifted.timestamp.tzinfo == timezone.utc

    with pytest.raises(ValidationError):
        resolution_response(started_at=datetime(2026, 7, 5, 12, 0))


def test_resolution_response_converged_matches_status():
    resolution_response(status="SUCCESS", converged=True)

    with pytest.raises(ValidationError):
        resolution_response(status="SUCCESS", converged=False)
    with pytest.raises(ValidationError):
        resolution_response(status="BUDGET_EXHAUSTED", converged=True, quality=None)


def test_response_models_reject_extra_fields():
    with pytest.raises(ValidationError):
        trace_event(extra=True)
    with pytest.raises(ValidationError):
        ReasoningEntry(iteration=0, role="agent", text="reasoning", extra=True)


def test_execution_trace_reasoning_defaults_to_empty_list():
    assert execution_trace().reasoning == []


def test_historical_response_without_version_stamp_remains_readable():
    stamped = resolution_response()
    payload = stamped.model_dump(
        exclude={
            "dataset_version",
            "solver_version",
            "action_policy_version",
            "ranking_policy_version",
            "result_schema_version",
        }
    )

    historical = ResolutionResponse.model_validate(payload)

    assert historical.dataset_version is None
    assert historical.solver_version is None
    assert historical.action_policy_version is None
    assert historical.ranking_policy_version is None
    assert historical.result_schema_version is None


def test_response_rejects_partial_version_stamp():
    with pytest.raises(ValidationError, match="version stamp"):
        resolution_response(dataset_version=None)
