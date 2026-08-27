# ABOUTME: Verifies restorebench.schemas public re-exports and cross-schema safety properties.
# ABOUTME: Round-trips every public BaseModel and guards against net/DataFrame/ndarray fields.
from datetime import datetime, timezone
from types import UnionType
from typing import get_origin
from uuid import UUID, uuid4

from pydantic import BaseModel

import restorebench.schemas as schemas


def round_trip(model: BaseModel) -> None:
    assert type(model).model_validate(model.model_dump()) == model
    assert type(model).model_validate_json(model.model_dump_json()) == model


def action() -> schemas.GenVoltageSetpointAction:
    return schemas.GenVoltageSetpointAction(type="GEN_V_SETPOINT", gen_id=1, new_vm_pu=1.02)


def maneuver() -> schemas.Maneuver:
    return schemas.Maneuver(action=action(), diagnosed_cause="REACTIVE_DEFICIT", rationale="Raise voltage.")


def diagnostics() -> schemas.NRDiagnostics:
    return schemas.NRDiagnostics(
        iterations_attempted=30,
        worst_bus=4,
        lowest_vm_pu=0.8,
        lowest_vm_bus=4,
        gens_at_q_limit=[],
        max_mismatch_mw=None,
        max_mismatch_mvar=None,
        overstress=None,
        error_message="failed",
        diagnostics_source="local_nose",
    )


def quality() -> schemas.QualityResult:
    return schemas.QualityResult(clean=True, n_buses_out_of_band=0, worst_vm_pu=0.99, worst_vm_bus=4, symptoms=[])


def gen_info() -> schemas.GenInfo:
    return schemas.GenInfo(
        gen_id=1,
        bus=4,
        p_mw=50.0,
        vm_pu=1.01,
        min_p_mw=0.0,
        max_p_mw=100.0,
        min_q_mvar=-40.0,
        max_q_mvar=40.0,
        dispatchable=True,
        voltage_control_status="PV_CONTROLLABLE",
        in_service=True,
    )


def topology() -> schemas.TopologySummary:
    return schemas.TopologySummary(
        n_buses=1,
        n_lines=1,
        n_trafos=1,
        n_gens=1,
        n_ext_grids=1,
        n_loads=1,
        n_shunts=1,
        buses=[schemas.BusInfo(bus_id=4, name=None, vn_kv=138.0, in_service=True)],
        lines=[schemas.LineInfo(line_id=0, from_bus=4, to_bus=5, in_service=True)],
        trafos=[schemas.TrafoInfo(trafo_id=0, hv_bus=4, lv_bus=5, tap_pos=0, tap_min=-2, tap_max=2, in_service=True)],
        gens=[gen_info()],
        ext_grids=[
            schemas.ExtGridInfo(
                ext_grid_id=0,
                bus=4,
                min_p_mw=-805.0,
                max_p_mw=805.0,
                min_q_mvar=-300.0,
                max_q_mvar=300.0,
                in_service=True,
            )
        ],
        loads=[schemas.LoadInfo(load_id=0, bus=4, p_mw=10.0, q_mvar=4.0, in_service=True)],
        shunts=[
            schemas.ShuntInfo(
                shunt_id=0,
                bus=4,
                q_mvar=-20.0,
                type="capacitor",
                step=1,
                max_step=1,
                in_service=True,
            )
        ],
        slack_bus=4,
    )


def trace_event() -> schemas.TraceEvent:
    return schemas.TraceEvent(
        timestamp=datetime(2026, 7, 5, 12, 0, tzinfo=timezone.utc),
        phase="solve",
        event_name="run_ac_pf",
        duration_ms=10.0,
        payload={},
    )


def execution_trace(request_id: UUID) -> schemas.ExecutionTrace:
    return schemas.ExecutionTrace(
        request_id=request_id,
        events=[trace_event()],
        n_llm_calls=0,
        total_llm_tokens_in=0,
        total_llm_tokens_out=0,
        n_tool_calls=1,
        n_power_flows=1,
    )


def all_model_examples() -> list[BaseModel]:
    request_id = uuid4()
    pf_result = schemas.PowerFlowResult(
        converged=True,
        iterations=7,
        tolerance_used=1e-8,
        runtime_ms=15.0,
        error_message=None,
        diagnostics=None,
        quality=quality(),
        warnings=[],
    )
    assignment = schemas.LLMAssignment(single_agent="qwen3:4b", analyst=None, executor=None, orchestrator=None)
    return [
        action(),
        schemas.ShuntStepAction(type="SHUNT_STEP", shunt_id=0, new_step=0),
        schemas.TapAdjustmentAction(type="TAP_ADJUSTMENT", trafo_id=0, new_tap_pos=1),
        maneuver(),
        schemas.ManeuverSequence(maneuvers=[maneuver()], reconstruction_summary=None),
        diagnostics(),
        schemas.QualitySymptom(type="V_LOW", element_id=4, value=0.94, threshold=0.95),
        schemas.OperatingWarning(
            type="GEN_Q_LIMIT",
            gen_id=1,
            q_mvar=40.0,
            nearest_bound=40.0,
            voltage_control_status="Q_LIMITED_UPPER",
        ),
        quality(),
        pf_result,
        schemas.ConvergenceResult(converged=True, iterations=7, tolerance_used=1e-8, diagnostics=None),
        schemas.BusInfo(bus_id=4, name=None, vn_kv=138.0, in_service=True),
        schemas.LineInfo(line_id=0, from_bus=4, to_bus=5, in_service=True),
        schemas.TrafoInfo(trafo_id=0, hv_bus=4, lv_bus=5, tap_pos=0, tap_min=-2, tap_max=2, in_service=True),
        gen_info(),
        schemas.ExtGridInfo(
            ext_grid_id=0, bus=4, min_p_mw=-805.0, max_p_mw=805.0, min_q_mvar=-300.0, max_q_mvar=300.0, in_service=True
        ),
        schemas.LoadInfo(load_id=0, bus=4, p_mw=10.0, q_mvar=4.0, in_service=True),
        schemas.ShuntInfo(
            shunt_id=0,
            bus=4,
            q_mvar=-20.0,
            type="capacitor",
            step=1,
            max_step=1,
            in_service=True,
        ),
        topology(),
        schemas.ApplicabilityResult(action=action(), applicable=True, reason=None),
        schemas.FailureFeedback(
            iteration=1, kind="STILL_DIVERGED", diagnostics=diagnostics(), detail=None, maneuver=maneuver()
        ),
        schemas.SandboxNet(sandbox_id=uuid4(), scenario_request_id=request_id),
        schemas.Scenario(
            scenario_id="S0001",
            full_net_path="dataset/ieee118/full/S0001.json",
            card_path="dataset/ieee118/llm/S0001.md",
            memory_split="memory_population",
        ),
        schemas.AnalystAssessment(
            diagnosed_cause="REACTIVE_DEFICIT", proposed_maneuver=maneuver(), rationale="x"
        ),
        schemas.ExecutorReport(
            maneuver=maneuver(),
            applicability=schemas.ApplicabilityResult(action=action(), applicable=True, reason=None),
            pf_result=pf_result,
        ),
        trace_event(),
        execution_trace(request_id),
        schemas.Citation(
            marker_id="${diag.lowest_vm_pu}",
            solver_field_path="iterations[0].diagnostics.lowest_vm_pu",
            resolved_value=0.9,
            formatting=None,
        ),
        schemas.ResolutionResponse(
            request_id=request_id,
            scenario_id="S0001",
            configuration=2,
            llm_assignment={"single_agent": "qwen3:4b"},
            repetition_index=3,
            status="SUCCESS",
            maneuvers=[maneuver()],
            n_maneuvers=1,
            converged=True,
            quality=quality(),
            final_warnings=[],
            diagnosis_rationale=None,
            citations=[],
            failure_feedback=[],
            trace=execution_trace(request_id),
            total_runtime_seconds=5.0,
            started_at=datetime(2026, 7, 5, 12, 0, tzinfo=timezone.utc),
            completed_at=datetime(2026, 7, 5, 12, 1, tzinfo=timezone.utc),
        ),
        assignment,
        schemas.OrchestratorConfig(CONFIGURATION=2, LLM_ASSIGNMENT=assignment, repetition_index=3),
    ]


def test_all_public_models_import_from_backend_schemas_and_round_trip():
    for model in all_model_examples():
        round_trip(model)


def test_no_schema_field_uses_pandapower_dataframe_or_ndarray_annotations():
    forbidden_fragments = ("pandapower", "DataFrame", "ndarray")
    for model_type in {type(model) for model in all_model_examples()}:
        for field in model_type.model_fields.values():
            assert not any(fragment in repr(field.annotation) for fragment in forbidden_fragments)


def test_public_model_set_matches_expected_exports():
    exported_models = {
        name
        for name in schemas.__all__
        if isinstance(getattr(schemas, name), type) and issubclass(getattr(schemas, name), BaseModel)
    }
    expected = {type(model).__name__ for model in all_model_examples()}
    assert expected <= exported_models


def test_action_union_export_is_not_a_model_or_net_annotation():
    origin = get_origin(schemas.Action)
    assert origin is not None or isinstance(schemas.Action, UnionType)
    assert "pandapower" not in repr(schemas.Action)
