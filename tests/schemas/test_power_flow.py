# ABOUTME: Verifies pure power-flow result schemas.
# ABOUTME: Covers diagnostics, quality, warnings, result variants, round trips, and extra rejection.
import pytest
from pydantic import ValidationError

from restorebench.schemas.power_flow import (
    ConvergenceResult,
    NRDiagnostics,
    OperatingWarning,
    PowerFlowResult,
    QualityResult,
    QualitySymptom,
)


def round_trip(model):
    assert type(model).model_validate(model.model_dump()) == model
    assert type(model).model_validate_json(model.model_dump_json()) == model


def diagnostics() -> NRDiagnostics:
    return NRDiagnostics(
        iterations_attempted=30,
        worst_bus=44,
        lowest_vm_pu=0.721,
        lowest_vm_bus=44,
        gens_at_q_limit=[1, 2],
        max_mismatch_mw=None,
        max_mismatch_mvar=None,
        overstress=0.12,
        error_message="load flow did not converge",
        diagnostics_source="local_nose",
    )


def quality() -> QualityResult:
    return QualityResult(
        clean=False,
        n_buses_out_of_band=1,
        worst_vm_pu=0.94,
        worst_vm_bus=10,
        symptoms=[QualitySymptom(type="V_LOW", element_id=10, value=0.94, threshold=0.95)],
    )


def warning() -> OperatingWarning:
    return OperatingWarning(
        type="GEN_Q_LIMIT",
        gen_id=3,
        q_mvar=40.0,
        nearest_bound=40.0,
        voltage_control_status="Q_LIMITED_UPPER",
    )


def test_power_flow_models_round_trip():
    models = [
        diagnostics(),
        QualitySymptom(type="V_HIGH", element_id=9, value=1.06, threshold=1.05),
        warning(),
        quality(),
        PowerFlowResult(
            converged=False,
            iterations=30,
            tolerance_used=1e-8,
            runtime_ms=12.5,
            error_message="failed",
            diagnostics=diagnostics(),
            quality=None,
            warnings=[warning()],
        ),
        ConvergenceResult(converged=False, iterations=30, tolerance_used=1e-8, diagnostics=diagnostics()),
    ]

    for model in models:
        round_trip(model)


def test_power_flow_models_reject_extra_fields():
    with pytest.raises(ValidationError):
        NRDiagnostics(
            iterations_attempted=30,
            worst_bus=44,
            lowest_vm_pu=0.721,
            lowest_vm_bus=44,
            gens_at_q_limit=[],
            error_message="failed",
            diagnostics_source="ppc_last_iterate",
            unexpected=True,
        )
