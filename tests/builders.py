# ABOUTME: Shared synthetic builders for the eval-layer and analysis tests.
# ABOUTME: Mirrors the fixed test topology/diagnostics/response patterns of tests/tools/test_memory.py.
from datetime import datetime, timezone
from uuid import uuid4

from restorebench.schemas.actions import Maneuver
from restorebench.schemas.power_flow import NRDiagnostics, QualityResult
from restorebench.schemas.response import ExecutionTrace, ResolutionResponse, TraceEvent
from restorebench.schemas.topology import (
    BusInfo,
    ExtGridInfo,
    GenInfo,
    LineInfo,
    LoadInfo,
    ShuntInfo,
    TopologySummary,
    TrafoInfo,
)


def topology() -> TopologySummary:
    buses = [
        BusInfo(bus_id=bus_id, name=f"bus {bus_id}", vn_kv=110.0, in_service=True)
        for bus_id in range(1, 7)
    ]
    lines = [
        LineInfo(line_id=1, from_bus=1, to_bus=2, in_service=True),
        LineInfo(line_id=2, from_bus=2, to_bus=3, in_service=True),
        LineInfo(line_id=3, from_bus=3, to_bus=4, in_service=True),
        LineInfo(line_id=4, from_bus=3, to_bus=5, in_service=True),
        LineInfo(line_id=5, from_bus=2, to_bus=6, in_service=True),
    ]
    trafos = [
        TrafoInfo(
            trafo_id=30,
            hv_bus=2,
            lv_bus=3,
            tap_pos=0,
            tap_min=-2,
            tap_max=2,
            in_service=True,
        )
    ]
    gens = [
        GenInfo(
            gen_id=10,
            bus=4,
            p_mw=80.0,
            vm_pu=1.01,
            min_p_mw=20.0,
            max_p_mw=120.0,
            min_q_mvar=-40.0,
            max_q_mvar=80.0,
            dispatchable=True,
            voltage_control_status="PV_CONTROLLABLE",
            in_service=True,
        ),
        GenInfo(
            gen_id=11,
            bus=6,
            p_mw=60.0,
            vm_pu=1.00,
            min_p_mw=60.0,
            max_p_mw=60.0,
            min_q_mvar=-20.0,
            max_q_mvar=30.0,
            dispatchable=False,
            voltage_control_status="Q_LIMITED_UPPER",
            in_service=True,
        ),
        GenInfo(
            gen_id=12,
            bus=1,
            p_mw=90.0,
            vm_pu=1.02,
            min_p_mw=10.0,
            max_p_mw=150.0,
            min_q_mvar=-50.0,
            max_q_mvar=90.0,
            dispatchable=True,
            voltage_control_status="PV_CONTROLLABLE",
            in_service=True,
        ),
    ]
    shunts = [
        ShuntInfo(
            shunt_id=20,
            bus=5,
            q_mvar=-12.0,
            type="capacitor",
            step=0,
            max_step=1,
            in_service=True,
        ),
        ShuntInfo(
            shunt_id=21,
            bus=2,
            q_mvar=8.0,
            type="reactor",
            step=1,
            max_step=1,
            in_service=True,
        ),
    ]
    loads = [LoadInfo(load_id=40, bus=3, p_mw=95.0, q_mvar=42.0, in_service=True)]
    return TopologySummary(
        n_buses=len(buses),
        n_lines=len(lines),
        n_trafos=len(trafos),
        n_gens=len(gens),
        n_ext_grids=1,
        n_loads=len(loads),
        n_shunts=len(shunts),
        buses=buses,
        lines=lines,
        trafos=trafos,
        gens=gens,
        ext_grids=[
            ExtGridInfo(
                ext_grid_id=1,
                bus=1,
                min_p_mw=-999.0,
                max_p_mw=999.0,
                min_q_mvar=-999.0,
                max_q_mvar=999.0,
                in_service=True,
            )
        ],
        loads=loads,
        shunts=shunts,
        slack_bus=1,
    )


def diagnostics(**overrides) -> NRDiagnostics:
    data = {
        "iterations_attempted": 30,
        "worst_bus": 3,
        "lowest_vm_pu": 0.62,
        "lowest_vm_bus": 3,
        "gens_at_q_limit": [11],
        "max_mismatch_mw": 4.0,
        "max_mismatch_mvar": 10.0,
        "overstress": 0.0,
        "error_message": "power flow did not converge",
        "diagnostics_source": "local_nose",
    }
    data.update(overrides)
    return NRDiagnostics(**data)


def quality(**overrides) -> QualityResult:
    data = {
        "clean": True,
        "n_buses_out_of_band": 0,
        "worst_vm_pu": 0.985,
        "worst_vm_bus": 3,
        "symptoms": [],
    }
    data.update(overrides)
    return QualityResult(**data)


def maneuver(action: dict, cause: str = "REACTIVE_DEFICIT") -> Maneuver:
    return Maneuver(action=action, diagnosed_cause=cause, rationale="synthetic maneuver")


def trace_with_diagnostics(diag: NRDiagnostics) -> ExecutionTrace:
    return ExecutionTrace(
        request_id=uuid4(),
        events=[
            TraceEvent(
                timestamp=datetime(2026, 7, 5, 12, 0, tzinfo=timezone.utc),
                phase="solve",
                event_name="nr_diagnostics",
                duration_ms=1.0,
                payload={"diagnostics": diag.model_dump()},
            )
        ],
        n_llm_calls=1,
        total_llm_tokens_in=10,
        total_llm_tokens_out=5,
        n_tool_calls=2,
        n_power_flows=3,
    )


def response(
    *,
    scenario_id: str,
    maneuvers: list[Maneuver],
    diag: NRDiagnostics | None = None,
    status: str = "SUCCESS",
    converged: bool = True,
    result_quality: QualityResult | None = None,
    configuration: int = 3,
) -> ResolutionResponse:
    return ResolutionResponse(
        request_id=uuid4(),
        scenario_id=scenario_id,
        configuration=configuration,
        llm_assignment={"analyst": "qwen3:4b", "executor": "qwen3:4b", "orchestrator": "qwen3:4b"},
        repetition_index=0,
        status=status,
        maneuvers=maneuvers,
        n_maneuvers=len(maneuvers),
        converged=converged,
        quality=result_quality,
        final_warnings=[],
        diagnosis_rationale=None,
        citations=[],
        failure_feedback=[],
        trace=trace_with_diagnostics(diag or diagnostics()),
        total_runtime_seconds=2.0,
        started_at=datetime(2026, 7, 5, 12, 0, tzinfo=timezone.utc),
        completed_at=datetime(2026, 7, 5, 12, 1, tzinfo=timezone.utc),
    )
