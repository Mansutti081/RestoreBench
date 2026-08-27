# ABOUTME: Measures observed solver-convergence transitions with a mandatory coarse scan before refinement.
# ABOUTME: Retains feasibility and solver accounting without claiming global monotonicity or a physical nose.
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from restorebench.physics.feasibility import evaluate_solved_feasibility
from restorebench.physics.policies import BOUNDARY_POLICY_VERSION
from restorebench.physics.solver import solve_locked_probe
from restorebench.schemas.physics import (
    BoundaryFeasibilityPolicy,
    BoundaryMeasurement,
    BoundaryProbeRecord,
)


StateBuilder = Callable[[float], Any]


def measure_boundary(
    state_builder: StateBuilder,
    *,
    coarse_coordinates: Sequence[float],
    refinement_resolution: float,
    feasibility_policy: BoundaryFeasibilityPolicy,
) -> BoundaryMeasurement:
    """Scan a frozen coordinate grid, then refine only its first clean solved/unsolved bracket."""
    coordinates = tuple(float(value) for value in coarse_coordinates)
    _validate_coordinates(coordinates, refinement_resolution)
    records: list[BoundaryProbeRecord] = []
    coarse_terminal_status: str | None = None

    for coordinate in coordinates:
        record = _evaluate_coordinate(state_builder, coordinate, phase="COARSE")
        records.append(record)
        terminal_status = _terminal_infeasibility(record, feasibility_policy)
        if terminal_status is not None:
            coarse_terminal_status = terminal_status
            break

    coarse_logical = [record.logical_result for record in records]
    if _observed_solved_unsolved_solved(coarse_logical):
        return _measurement("OBSERVED_NON_MONOTONIC", None, None, records)

    bracket = _first_clean_transition(records)
    if bracket is None:
        if coarse_terminal_status is not None:
            return _measurement(coarse_terminal_status, None, None, records)
        return _measurement("NO_TRANSITION", None, None, records)
    lower, upper = bracket

    while upper - lower > refinement_resolution:
        midpoint = (lower + upper) / 2.0
        record = _evaluate_coordinate(state_builder, midpoint, phase="REFINEMENT")
        records.append(record)
        terminal_status = _terminal_infeasibility(record, feasibility_policy)
        if terminal_status is not None:
            return _measurement(terminal_status, None, None, records)
        if record.logical_result == "SOLVED":
            lower = midpoint
        elif record.logical_result == "NO_SOLUTION":
            upper = midpoint
        else:
            raise AssertionError("unexpected refinement result")

    return _measurement("BOUNDARY_FOUND", lower, upper, records)


def _evaluate_coordinate(
    state_builder: StateBuilder,
    coordinate: float,
    *,
    phase: str,
) -> BoundaryProbeRecord:
    state = state_builder(coordinate)
    active_balance = getattr(state, "active_balance", None)
    if active_balance is not None and active_balance.status == "ACTIVE_HEADROOM_EXHAUSTED":
        return BoundaryProbeRecord(
            coordinate=coordinate,
            logical_result="NOT_RUN",
            probe_status="INFEASIBLE",
            feasible=False,
            infeasibility_codes=("ACTIVE_HEADROOM_EXHAUSTED",),
            logical_probe_count=0,
            solver_attempt_count=0,
            phase=phase,
        )

    net = getattr(state, "net", state)
    probe = solve_locked_probe(net)
    if probe.status == "NO_SOLUTION":
        return BoundaryProbeRecord(
            coordinate=coordinate,
            logical_result="NO_SOLUTION",
            probe_status="NO_SOLUTION",
            feasible=None,
            logical_probe_count=1,
            solver_attempt_count=probe.solver_attempt_count,
            phase=phase,
        )

    feasibility = evaluate_solved_feasibility(probe.solved_net)
    if not feasibility.feasible:
        codes = tuple(reason.code for reason in feasibility.failure_reasons)
        return BoundaryProbeRecord(
            coordinate=coordinate,
            logical_result="SOLVED",
            probe_status="INFEASIBLE",
            feasible=False,
            infeasibility_codes=codes,
            logical_probe_count=1,
            solver_attempt_count=probe.solver_attempt_count,
            phase=phase,
        )
    return BoundaryProbeRecord(
        coordinate=coordinate,
        logical_result="SOLVED",
        probe_status="SOLVED",
        feasible=True,
        logical_probe_count=1,
        solver_attempt_count=probe.solver_attempt_count,
        phase=phase,
    )


def _terminal_infeasibility(
    record: BoundaryProbeRecord,
    policy: BoundaryFeasibilityPolicy,
) -> str | None:
    if record.probe_status != "INFEASIBLE":
        return None
    if "ACTIVE_HEADROOM_EXHAUSTED" in record.infeasibility_codes:
        return "ACTIVE_HEADROOM_EXHAUSTED"
    if any(code.startswith("EXT_GRID_") for code in record.infeasibility_codes):
        return "SLACK_INFEASIBLE"
    voltage_only = set(record.infeasibility_codes) == {
        "HARD_VOLTAGE_ENVELOPE"
    }
    if voltage_only and not policy.stop_on_solved_infeasibility:
        return None
    return "SOLVED_STATE_INFEASIBLE"


def _observed_solved_unsolved_solved(statuses: Sequence[str]) -> bool:
    seen_solved = False
    seen_unsolved_after_solved = False
    for status in statuses:
        if status == "SOLVED":
            if seen_unsolved_after_solved:
                return True
            seen_solved = True
        elif status == "NO_SOLUTION" and seen_solved:
            seen_unsolved_after_solved = True
    return False


def _first_clean_transition(
    records: Sequence[BoundaryProbeRecord],
) -> tuple[float, float] | None:
    for lower, upper in zip(records, records[1:], strict=False):
        if (
            lower.logical_result == "SOLVED"
            and upper.logical_result == "NO_SOLUTION"
        ):
            return lower.coordinate, upper.coordinate
    return None


def _measurement(
    status: str,
    highest_solved: float | None,
    lowest_unsolved: float | None,
    records: Sequence[BoundaryProbeRecord],
) -> BoundaryMeasurement:
    return BoundaryMeasurement(
        status=status,
        highest_solved=highest_solved,
        lowest_unsolved=lowest_unsolved,
        records=tuple(records),
        logical_probe_count=sum(record.logical_probe_count for record in records),
        solver_attempt_count=sum(record.solver_attempt_count for record in records),
        policy_version=BOUNDARY_POLICY_VERSION,
    )


def _validate_coordinates(coordinates: tuple[float, ...], resolution: float) -> None:
    if len(coordinates) < 2:
        raise ValueError("coarse coordinate grid must contain at least two values")
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("coarse coordinates must be finite")
    if any(right <= left for left, right in zip(coordinates, coordinates[1:], strict=False)):
        raise ValueError("coarse coordinates must be strictly increasing")
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("refinement_resolution must be finite and positive")
