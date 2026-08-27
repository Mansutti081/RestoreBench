# ABOUTME: Verifies a run_ac_pf preview reports its effect on overstress, not just a bare number.
# ABOUTME: A preview that closes a fifth of the gap must not read the same as one that widens it.
from __future__ import annotations

from typing import Any

import pytest

from restorebench.agents import tool_loop
from restorebench.agents.progress import overstress_verdict
from restorebench.schemas.power_flow import NRDiagnostics, PowerFlowResult
from restorebench.tools.sandbox import create_sandbox, discard_sandbox
from restorebench.corpus.augment import build_augmented_base


def _diagnostics(overstress: float | None) -> NRDiagnostics:
    return NRDiagnostics(
        iterations_attempted=30,
        worst_bus=35,
        lowest_vm_pu=0.61,
        lowest_vm_bus=35,
        gens_at_q_limit=[6, 7],
        overstress=overstress,
        error_message="did not converge",
        diagnostics_source="local_nose",
    )


def _diverged(overstress: float | None) -> PowerFlowResult:
    return PowerFlowResult(
        converged=False,
        iterations=30,
        tolerance_used=1e-6,
        runtime_ms=1.0,
        error_message="did not converge",
        diagnostics=_diagnostics(overstress),
    )


def _converged() -> PowerFlowResult:
    return PowerFlowResult(converged=True, iterations=6, tolerance_used=1e-8, runtime_ms=1.0)


def test_verdict_names_the_direction_of_travel() -> None:
    assert overstress_verdict(0.02958, 0.02397) == "improved"
    assert overstress_verdict(0.02397, 0.02958) == "worsened"
    assert overstress_verdict(0.02958, 0.02958) == "unchanged"


def test_verdict_is_unknown_when_either_reading_is_missing() -> None:
    """A converged-but-infeasible state has no nose diagnostics; that is not 'unchanged'."""
    assert overstress_verdict(None, 0.02958) == "unknown"
    assert overstress_verdict(0.02958, None) == "unknown"


def _preview(monkeypatch: pytest.MonkeyPatch, results: list[PowerFlowResult]) -> tuple[list[dict[str, Any]], int]:
    """Run one or more previews against a scripted solver and return their payloads."""
    net = build_augmented_base()
    n_previews = len(results) - 1
    candidates = []
    for gen_id in net.gen.index[net.gen["in_service"].astype(bool)]:
        vm_pu = float(net.gen.at[gen_id, "vm_pu"])
        if not 0.95 <= vm_pu < 1.05:
            continue
        candidates.append((int(gen_id), round(vm_pu + 0.01, 10)))
        if len(candidates) == n_previews:
            break
    assert len(candidates) == n_previews
    grid = create_sandbox(net)

    calls = {"n": 0}
    remaining = list(results)

    def fake_run_ac_pf(_net: Any, **_kwargs: Any) -> PowerFlowResult:
        calls["n"] += 1
        return remaining.pop(0)

    monkeypatch.setattr(tool_loop, "run_ac_pf", fake_run_ac_pf)
    handler = tool_loop.default_diagnostic_tools()[2].handler

    payloads = []
    try:
        for gen_id, target in candidates:
            maneuver = {
                "action": {"type": "GEN_V_SETPOINT", "gen_id": gen_id, "new_vm_pu": target},
                "diagnosed_cause": "REACTIVE_DEFICIT",
                "rationale": "probe",
            }
            payloads.append(handler(grid, {"maneuver": maneuver}))
    finally:
        discard_sandbox(grid)
    return payloads, calls["n"]


def test_preview_reports_both_readings_and_the_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads, _ = _preview(monkeypatch, [_diverged(0.02958), _diverged(0.02397)])

    payload = payloads[0]
    assert payload["overstress_before"] == pytest.approx(0.02958)
    assert payload["overstress_after"] == pytest.approx(0.02397)
    assert payload["overstress_verdict"] == "improved"


def test_a_preview_that_widens_the_gap_is_labelled_worsened(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads, _ = _preview(monkeypatch, [_diverged(0.02958), _diverged(0.04562)])

    assert payloads[0]["overstress_verdict"] == "worsened"


def test_the_baseline_is_solved_once_across_several_previews(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each solve costs seconds; re-measuring the unchanged grid per preview would double the bill."""
    _, n_calls = _preview(monkeypatch, [_diverged(0.02958), _diverged(0.02397), _diverged(0.03100)])

    # one baseline solve plus one per preview
    assert n_calls == 3


def _repeat_same_action(
    monkeypatch: pytest.MonkeyPatch,
    results: list[PowerFlowResult],
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Preview one action twice against a scripted solver, returning payloads and failed attempts."""
    net = build_augmented_base()
    gen_id = int(net.gen.index[net.gen["in_service"].astype(bool)][0])
    vm_pu = float(net.gen.at[gen_id, "vm_pu"])
    target = round(vm_pu + 0.01, 10) if vm_pu < 1.05 else round(vm_pu - 0.01, 10)
    grid = create_sandbox(net)
    remaining = list(results)

    def fake_run_ac_pf(_net: Any, **_kwargs: Any) -> PowerFlowResult:
        return remaining.pop(0)

    monkeypatch.setattr(tool_loop, "run_ac_pf", fake_run_ac_pf)
    attempts: list[Any] = []
    handler = tool_loop.default_diagnostic_tools(failed_attempts=attempts)[2].handler
    maneuver = {
        "action": {"type": "GEN_V_SETPOINT", "gen_id": gen_id, "new_vm_pu": target},
        "diagnosed_cause": "REACTIVE_DEFICIT",
        "rationale": "probe",
    }
    payloads = []
    try:
        for _ in range(2):
            payloads.append(handler(grid, {"maneuver": maneuver}))
    finally:
        discard_sandbox(grid)
    return payloads, attempts


def test_an_improving_preview_stays_available_to_propose(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blocking it would forbid the agent from committing the best move it found."""
    payloads, attempts = _repeat_same_action(
        monkeypatch, [_diverged(0.02958), _diverged(0.01570), _diverged(0.01570)]
    )

    assert payloads[0]["overstress_verdict"] == "improved"
    assert payloads[1]["overstress_verdict"] == "improved"
    assert attempts == []


def test_a_worsening_preview_is_still_blocked_from_repeating(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(tool_loop.InvalidActionError):
        _repeat_same_action(monkeypatch, [_diverged(0.02958), _diverged(0.04562), _diverged(0.04562)])


def test_an_unmeasurable_preview_is_blocked_because_progress_is_unproven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(tool_loop.InvalidActionError):
        _repeat_same_action(monkeypatch, [_diverged(0.02958), _diverged(None), _diverged(None)])


def test_a_converging_preview_is_not_labelled_by_overstress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convergence is the terminal signal; a verdict there would compete with it."""
    payloads, _ = _preview(monkeypatch, [_diverged(0.02958), _converged()])

    payload = payloads[0]
    assert payload["converged"] is True
    assert "overstress_verdict" not in payload


def test_a_change_at_the_measurement_quantum_is_not_progress() -> None:
    """22 bisection steps from 0.5 put the quantum at ~1.2e-7; that is the instrument, not the grid."""
    assert overstress_verdict(0.03329307871920362, 0.03329295144010613) == "unchanged"


def test_a_real_improvement_still_reads_as_progress_near_convergence() -> None:
    """The floor is relative, so it must not swallow genuine gains once overstress is small."""
    assert overstress_verdict(0.0016, 0.0012) == "improved"


def test_a_noise_level_worsening_is_also_flattened() -> None:
    assert overstress_verdict(0.03329295144010613, 0.03329307871920362) == "unchanged"


def test_the_ranking_is_computed_once_per_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    """It is deterministic on an unchanged grid, so a second call would buy twelve solves of nothing."""
    calls = {"n": 0}

    def fake_rank(*_a: Any, **_k: Any) -> Any:
        calls["n"] += 1
        return "ranking-object"

    monkeypatch.setattr(tool_loop, "rank_candidates", fake_rank)
    monkeypatch.setattr(tool_loop, "ranking_payload", lambda ranking: {"ranking": ranking})

    grid = create_sandbox(build_augmented_base())
    try:
        handler = tool_loop.default_diagnostic_tools()[3].handler
        first = handler(grid, {})
        second = handler(grid, {})
    finally:
        discard_sandbox(grid)

    assert first == second
    assert calls["n"] == 1
