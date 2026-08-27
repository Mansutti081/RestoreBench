# ABOUTME: Verifies scan-first convergence-boundary measurement and complete probe accounting.
# ABOUTME: Guards against bisection before a coarse bracket or certification of observed reversals.
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from restorebench.physics import boundary
from restorebench.physics.boundary import measure_boundary
from restorebench.schemas.physics import BoundaryFeasibilityPolicy


def _install_fake_physics(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status_at: Any,
    call_order: list[float],
) -> None:
    def fake_solver(net: dict[str, float]) -> Any:
        coordinate = net["coordinate"]
        call_order.append(coordinate)
        status = status_at(coordinate)
        return SimpleNamespace(
            status=status,
            solved_net=net if status == "SOLVED" else None,
            solver_attempt_count=1 if status == "SOLVED" else 2,
        )

    monkeypatch.setattr(boundary, "solve_locked_probe", fake_solver)
    monkeypatch.setattr(
        boundary,
        "evaluate_solved_feasibility",
        lambda _net: SimpleNamespace(feasible=True, failure_reasons=()),
    )


def _state_builder(coordinate: float) -> Any:
    return SimpleNamespace(
        net={"coordinate": coordinate},
        active_balance=SimpleNamespace(status="SCHEDULED"),
    )


def test_boundary_scans_full_coarse_grid_before_refining_first_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order: list[float] = []
    _install_fake_physics(
        monkeypatch,
        status_at=lambda coordinate: "SOLVED" if coordinate < 1.2 else "NO_SOLUTION",
        call_order=call_order,
    )

    result = measure_boundary(
        _state_builder,
        coarse_coordinates=(0.0, 1.0, 2.0),
        refinement_resolution=0.125,
        feasibility_policy=BoundaryFeasibilityPolicy(),
    )

    assert result.status == "BOUNDARY_FOUND"
    assert call_order[:3] == [0.0, 1.0, 2.0]
    assert result.highest_solved == pytest.approx(1.125)
    assert result.lowest_unsolved == pytest.approx(1.25)
    assert result.lowest_unsolved - result.highest_solved <= 0.125


def test_boundary_records_observed_solved_unsolved_solved_without_refinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order: list[float] = []
    statuses = {0.0: "SOLVED", 1.0: "NO_SOLUTION", 2.0: "SOLVED"}
    _install_fake_physics(
        monkeypatch,
        status_at=statuses.__getitem__,
        call_order=call_order,
    )

    result = measure_boundary(
        _state_builder,
        coarse_coordinates=(0.0, 1.0, 2.0),
        refinement_resolution=0.01,
        feasibility_policy=BoundaryFeasibilityPolicy(),
    )

    assert result.status == "OBSERVED_NON_MONOTONIC"
    assert call_order == [0.0, 1.0, 2.0]
    assert result.highest_solved is None
    assert result.lowest_unsolved is None


def test_boundary_stops_with_typed_active_headroom_exhaustion_before_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver_calls: list[float] = []
    _install_fake_physics(
        monkeypatch,
        status_at=lambda _coordinate: "SOLVED",
        call_order=solver_calls,
    )

    def builder(coordinate: float) -> Any:
        status = "ACTIVE_HEADROOM_EXHAUSTED" if coordinate == 1.0 else "SCHEDULED"
        return SimpleNamespace(
            net={"coordinate": coordinate},
            active_balance=SimpleNamespace(status=status),
        )

    result = measure_boundary(
        builder,
        coarse_coordinates=(0.0, 1.0, 2.0),
        refinement_resolution=0.1,
        feasibility_policy=BoundaryFeasibilityPolicy(),
    )

    assert result.status == "ACTIVE_HEADROOM_EXHAUSTED"
    assert solver_calls == [0.0]
    assert result.records[-1].coordinate == 1.0
    assert result.records[-1].probe_status == "INFEASIBLE"
    assert result.records[-1].solver_attempt_count == 0


def test_active_limit_beyond_a_clean_coarse_transition_does_not_hide_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver_calls: list[float] = []
    _install_fake_physics(
        monkeypatch,
        status_at=lambda coordinate: "SOLVED" if coordinate < 0.75 else "NO_SOLUTION",
        call_order=solver_calls,
    )

    def builder(coordinate: float) -> Any:
        status = "ACTIVE_HEADROOM_EXHAUSTED" if coordinate == 2.0 else "SCHEDULED"
        return SimpleNamespace(
            net={"coordinate": coordinate},
            active_balance=SimpleNamespace(status=status),
        )

    result = measure_boundary(
        builder,
        coarse_coordinates=(0.0, 1.0, 2.0),
        refinement_resolution=0.5,
        feasibility_policy=BoundaryFeasibilityPolicy(),
    )

    assert result.status == "BOUNDARY_FOUND"
    assert solver_calls[:2] == [0.0, 1.0]
    assert result.highest_solved == pytest.approx(0.5)
    assert result.lowest_unsolved == pytest.approx(1.0)


def test_boundary_accounting_sums_every_logical_probe_and_solver_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order: list[float] = []
    _install_fake_physics(
        monkeypatch,
        status_at=lambda coordinate: "SOLVED" if coordinate < 1.5 else "NO_SOLUTION",
        call_order=call_order,
    )

    result = measure_boundary(
        _state_builder,
        coarse_coordinates=(0.0, 1.0, 2.0),
        refinement_resolution=0.5,
        feasibility_policy=BoundaryFeasibilityPolicy(),
    )

    assert result.logical_probe_count == len(call_order)
    assert result.logical_probe_count == sum(record.logical_probe_count for record in result.records)
    assert result.solver_attempt_count == sum(record.solver_attempt_count for record in result.records)


def test_boundary_requires_ordered_coarse_grid_and_never_refines_without_bracket() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        measure_boundary(
            _state_builder,
            coarse_coordinates=(0.0, 2.0, 1.0),
            refinement_resolution=0.1,
            feasibility_policy=BoundaryFeasibilityPolicy(),
        )


def test_voltage_only_infeasible_probes_can_measure_solver_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order: list[float] = []
    _install_fake_physics(
        monkeypatch,
        status_at=lambda coordinate: "SOLVED" if coordinate < 1.75 else "NO_SOLUTION",
        call_order=call_order,
    )

    def fake_feasibility(net: dict[str, float]) -> Any:
        feasible = net["coordinate"] < 1.25
        return SimpleNamespace(
            feasible=feasible,
            failure_reasons=(
                ()
                if feasible
                else (SimpleNamespace(code="HARD_VOLTAGE_ENVELOPE"),)
            ),
        )

    monkeypatch.setattr(
        boundary,
        "evaluate_solved_feasibility",
        fake_feasibility,
    )

    result = measure_boundary(
        _state_builder,
        coarse_coordinates=(0.0, 1.0, 1.5, 2.0),
        refinement_resolution=0.125,
        feasibility_policy=BoundaryFeasibilityPolicy(
            stop_on_solved_infeasibility=False,
        ),
    )

    assert result.status == "BOUNDARY_FOUND"
    assert result.highest_solved == pytest.approx(1.625)
    assert result.lowest_unsolved == pytest.approx(1.75)
    assert any(
        record.logical_result == "SOLVED"
        and record.probe_status == "INFEASIBLE"
        for record in result.records
    )


def test_non_voltage_infeasibility_still_stops_solver_boundary_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order: list[float] = []
    _install_fake_physics(
        monkeypatch,
        status_at=lambda coordinate: "SOLVED" if coordinate < 2.0 else "NO_SOLUTION",
        call_order=call_order,
    )
    monkeypatch.setattr(
        boundary,
        "evaluate_solved_feasibility",
        lambda net: SimpleNamespace(
            feasible=net["coordinate"] < 1.0,
            failure_reasons=(
                ()
                if net["coordinate"] < 1.0
                else (SimpleNamespace(code="EXT_GRID_Q_LIMIT"),)
            ),
        ),
    )

    result = measure_boundary(
        _state_builder,
        coarse_coordinates=(0.0, 1.0, 2.0),
        refinement_resolution=0.1,
        feasibility_policy=BoundaryFeasibilityPolicy(
            stop_on_solved_infeasibility=False,
        ),
    )

    assert result.status == "SLACK_INFEASIBLE"
    assert call_order == [0.0, 1.0]
