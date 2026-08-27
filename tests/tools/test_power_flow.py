# ABOUTME: Verifies the deterministic PowerFlowServer tool against real IEEE-118 nets.
# ABOUTME: Covers locked solver settings, diagnostics, quality, warnings, sandboxing, and dataset scenarios.
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pandapower as pp
import pytest
from pandas.testing import assert_frame_equal
from pandapower.auxiliary import LoadflowNotConverged

from restorebench.corpus.augment import build_augmented_base


ROOT = Path(__file__).resolve().parents[2]
DATASET_FULL = ROOT / "dataset/ieee118/full"


def _dataframe_snapshot(net) -> dict[str, pd.DataFrame]:
    return {key: value.copy(deep=True) for key, value in net.items() if isinstance(value, pd.DataFrame)}


def _assert_net_tables_unchanged(before: dict[str, pd.DataFrame], net) -> None:
    assert set(before) == {key for key, value in net.items() if isinstance(value, pd.DataFrame)}
    for key, frame in before.items():
        assert_frame_equal(frame, net[key], check_exact=True)


def _load_full_scenario(scenario_id: str):
    return pp.from_json(str(DATASET_FULL / f"{scenario_id}.json"))


def _uniform_collapse_net(scale: float = 1.6):
    net = build_augmented_base()
    net.load[["p_mw", "q_mvar"]] *= scale
    pp.reset_results(net)
    return net


def test_run_ac_pf_converged_base_reports_quality_and_preserves_caller_net():
    from restorebench.tools.power_flow import run_ac_pf

    net = build_augmented_base()
    before = _dataframe_snapshot(net)

    result = run_ac_pf(net)

    assert result.converged is True
    assert result.diagnostics is None
    assert result.quality is not None
    assert result.quality.clean is False
    assert result.quality.n_buses_out_of_band > 0
    assert result.warnings
    assert result.tolerance_used == pytest.approx(1e-8)
    assert result.iterations > 0
    assert result.runtime_ms >= 0
    _assert_net_tables_unchanged(before, net)


def test_run_ac_pf_uniform_collapse_uses_local_nose_diagnostics():
    from restorebench.tools.power_flow import run_ac_pf

    result = run_ac_pf(_uniform_collapse_net())

    assert result.converged is False
    assert result.quality is None
    assert result.diagnostics is not None
    diagnostics = result.diagnostics
    assert diagnostics.diagnostics_source == "local_nose"
    assert diagnostics.iterations_attempted == 30
    assert diagnostics.lowest_vm_bus == 117
    assert diagnostics.worst_bus == diagnostics.lowest_vm_bus
    assert 0.5 < diagnostics.lowest_vm_pu < 0.75
    assert diagnostics.overstress is not None
    assert diagnostics.overstress > 0
    assert diagnostics.gens_at_q_limit
    assert diagnostics.max_mismatch_mw is None
    assert diagnostics.max_mismatch_mvar is None


def test_assess_quality_reports_low_and_high_voltage_only():
    from restorebench.tools.power_flow import assess_quality

    net = build_augmented_base()
    pp.runpp(net, algorithm="nr", enforce_q_lims=True, init="dc", tolerance_mva=1e-8, max_iteration=30)
    net.res_bus["vm_pu"] = 1.0
    net.res_bus.at[2, "vm_pu"] = 0.94
    net.res_bus.at[3, "vm_pu"] = 1.06
    if len(net.res_line):
        net.res_line["loading_percent"] = 999.0

    result = assess_quality(net)

    assert result.clean is False
    assert result.n_buses_out_of_band == 2
    assert {(symptom.type, symptom.element_id) for symptom in result.symptoms} == {
        ("V_LOW", 2),
        ("V_HIGH", 3),
    }


def test_detect_warnings_marks_upper_q_limit_directionally():
    from restorebench.tools.power_flow import detect_warnings

    net = build_augmented_base()
    pp.runpp(net, algorithm="nr", enforce_q_lims=True, init="dc", tolerance_mva=1e-8, max_iteration=30)
    gen_id = int(net.gen.index[0])
    max_q = float(net.gen.at[gen_id, "max_q_mvar"])
    net.res_gen.at[gen_id, "q_mvar"] = max_q

    warnings = detect_warnings(net)

    q_limit = next(warning for warning in warnings if warning.gen_id == gen_id)
    assert q_limit.type == "GEN_Q_LIMIT"
    assert q_limit.nearest_bound == pytest.approx(max_q)
    assert q_limit.voltage_control_status == "Q_LIMITED_UPPER"


def test_locked_power_flow_keeps_generator_q_within_declared_limits():
    from restorebench.tools.power_flow import DEFAULT_TOLERANCE, Q_LIMIT_EPS, _run_locked_pf

    net = build_augmented_base()

    _run_locked_pf(net, DEFAULT_TOLERANCE)

    in_service = net.gen["in_service"].astype(bool)
    q_mvar = net.res_gen.loc[in_service, "q_mvar"]
    min_q_mvar = net.gen.loc[in_service, "min_q_mvar"]
    max_q_mvar = net.gen.loc[in_service, "max_q_mvar"]
    assert (q_mvar >= min_q_mvar - Q_LIMIT_EPS).all()
    assert (q_mvar <= max_q_mvar + Q_LIMIT_EPS).all()


def test_q_limit_diagnostics_only_mark_upper_saturation_for_raise_guard():
    from restorebench.tools.power_flow import _gens_at_q_limit_from_results

    net = build_augmented_base()
    pp.runpp(net, algorithm="nr", enforce_q_lims=True, init="dc", tolerance_mva=1e-8, max_iteration=30)
    lower_gen, upper_gen = (int(gen_id) for gen_id in net.gen.index[:2])
    net.res_gen.at[lower_gen, "q_mvar"] = float(net.gen.at[lower_gen, "min_q_mvar"])
    net.res_gen.at[upper_gen, "q_mvar"] = float(net.gen.at[upper_gen, "max_q_mvar"])

    saturated_gens = _gens_at_q_limit_from_results(net)

    assert lower_gen not in saturated_gens
    assert upper_gen in saturated_gens


def test_auto_recovery_reports_relaxed_tolerance(monkeypatch):
    import restorebench.tools.power_flow as power_flow
    from restorebench.physics import solver

    net = build_augmented_base()
    original = solver._run_locked_pf
    calls: list[float] = []

    def fail_tight_once(net, tolerance_mva: float) -> None:
        calls.append(tolerance_mva)
        if tolerance_mva == power_flow.DEFAULT_TOLERANCE:
            raise LoadflowNotConverged("forced tight tolerance failure")
        original(net, tolerance_mva)

    monkeypatch.setattr(solver, "_run_locked_pf", fail_tight_once)

    result = power_flow.run_ac_pf(net)

    assert calls == [power_flow.DEFAULT_TOLERANCE, power_flow.RECOVERY_TOLERANCE]
    assert result.converged is True
    assert result.tolerance_used == pytest.approx(power_flow.RECOVERY_TOLERANCE)


@pytest.mark.parametrize("scenario_id", ['S0008', 'S0012', 'S0014', 'S0016', 'S0019', 'S0029', 'S0039', 'S0041', 'S0046', 'S0048'])
def test_dataset_scenarios_report_nonconvergence_with_usable_diagnostics(scenario_id: str):
    from restorebench.tools.power_flow import run_ac_pf

    net = _load_full_scenario(scenario_id)

    result = run_ac_pf(net)

    assert result.converged is False
    assert result.quality is None
    assert result.diagnostics is not None
    assert result.diagnostics.diagnostics_source == "local_nose"
    assert pd.notna(result.diagnostics.lowest_vm_pu)
    assert result.diagnostics.lowest_vm_pu > 0
    assert result.diagnostics.overstress is not None
    assert result.diagnostics.overstress > 0
    assert result.diagnostics.gens_at_q_limit


def test_sandboxnet_inputs_are_resolved_via_sandbox_server():
    from restorebench.tools import sandbox as SandboxServer
    from restorebench.tools.power_flow import run_ac_pf

    source_net = build_augmented_base()
    before = _dataframe_snapshot(source_net)
    handle = SandboxServer.create_sandbox(source_net)

    try:
        result = run_ac_pf(handle)
    finally:
        SandboxServer.discard_sandbox(handle)

    assert result.converged is True
    _assert_net_tables_unchanged(before, source_net)


def test_runtime_diagnostics_read_the_shared_retreat_when_it_yields_evidence(monkeypatch):
    # The applicability guard blocks raising a Q-saturated generator, so a runtime Q context
    # read at a different retreat point saturates generators the corpus witness considers
    # controllable - forbidding the agent from ever applying that scenario's witness action.
    from restorebench.physics.feasibility import evaluate_solved_feasibility
    from restorebench.physics.retreat import RetreatEvidence
    from restorebench.tools import power_flow as power_flow_module

    solved = build_augmented_base()
    pp.runpp(solved, algorithm="nr", enforce_q_lims=True, init="dc", max_iteration=30)
    real_status = evaluate_solved_feasibility(solved).generator_q_status
    shared_upper = [int(real_status[0].gen_id), int(real_status[2].gen_id)]
    evidence = RetreatEvidence(
        lambda_value=0.9,
        solved_net=solved,
        q_status=tuple(
            item.model_copy(
                update={
                    "status": "Q_LIMITED_UPPER" if int(item.gen_id) in shared_upper else "PV_CONTROLLABLE"
                }
            )
            for item in real_status
        ),
        logical_probe_count=4,
        solver_attempt_count=5,
    )
    monkeypatch.setattr(power_flow_module, "retreat_evidence", lambda _net: evidence)

    diagnostics = power_flow_module.run_ac_pf(_uniform_collapse_net()).diagnostics

    assert diagnostics.gens_at_q_limit == shared_upper
    assert diagnostics.overstress == pytest.approx((1.0 / 0.9) - 1.0)


def test_runtime_diagnostics_fall_back_when_the_shared_retreat_has_no_evidence(monkeypatch):
    # A corpus built on another trajectory legitimately yields no shared evidence; diagnostics
    # must still be produced rather than failing the scenario.
    from restorebench.tools import power_flow as power_flow_module

    monkeypatch.setattr(power_flow_module, "retreat_evidence", lambda _net: None)

    diagnostics = power_flow_module.run_ac_pf(_uniform_collapse_net()).diagnostics

    assert diagnostics.diagnostics_source == "local_nose"
    assert diagnostics.lowest_vm_pu > 0.0


def test_active_diagnostics_reject_failed_iterate_result_tables():
    from restorebench.tools.power_flow import extract_nr_diagnostics

    failed_net = SimpleNamespace(_ppc={})

    with pytest.raises(ValueError, match="Unknown diagnostics source"):
        extract_nr_diagnostics(
            failed_net,
            error_message="did not converge",
            diagnostics_source="ppc_last_iterate",  # type: ignore[arg-type]
        )


def test_shared_retreat_absence_is_a_failure_unless_the_legacy_route_is_requested(monkeypatch):
    # A pre-refactor snapshot legitimately yields no shared evidence, but a target snapshot doing
    # the same is a regression that would silently restore the runtime/witness Q-context split.
    # Falling back by default would hide exactly the failure the shared retreat exists to prevent.
    from restorebench.schemas.errors import ToolFailureError
    from restorebench.tools import power_flow as power_flow_module

    monkeypatch.setattr(power_flow_module, "retreat_evidence", lambda _net: None)
    net = _uniform_collapse_net()

    with pytest.raises(ToolFailureError, match="legacy retreat"):
        power_flow_module.extract_nr_diagnostics(
            failed_net=net,
            source_net=net,
            error_message="did not converge",
            allow_legacy_retreat=False,
        )

    with pytest.warns(power_flow_module.LegacyRetreatWarning):
        diagnostics = power_flow_module.extract_nr_diagnostics(
            failed_net=net,
            source_net=net,
            error_message="did not converge",
            allow_legacy_retreat=True,
        ).lowest_vm_pu

    assert diagnostics > 0.0

