# ABOUTME: Derives generator Q status from the shared snapshot-anchored retreat point.
# ABOUTME: Keeps runtime diagnostics and corpus witness search on one retreat contract.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from restorebench.physics.active_balance import build_active_policy
from restorebench.physics.boundary import measure_boundary
from restorebench.physics.feasibility import evaluate_solved_feasibility
from restorebench.physics.policies import (
    RETREAT_COORDINATES,
    RETREAT_POLICY_VERSION,
    RETREAT_RESOLUTION,
)
from restorebench.physics.solver import solve_locked_probe
from restorebench.physics.trajectory import build_diagnostic_state
from restorebench.schemas.physics import BoundaryFeasibilityPolicy, GeneratorQState


@dataclass(frozen=True)
class RetreatEvidence:
    """Solved, feasible evidence taken at the shared snapshot-anchored retreat point."""

    lambda_value: float
    solved_net: Any
    q_status: tuple[GeneratorQState, ...]
    logical_probe_count: int
    solver_attempt_count: int
    policy_version: str = RETREAT_POLICY_VERSION

    @property
    def upper_q_limited_gen_ids(self) -> tuple[int, ...]:
        return tuple(
            sorted(int(item.gen_id) for item in self.q_status if item.status == "Q_LIMITED_UPPER")
        )


def retreat_evidence(
    snapshot_net: Any,
    *,
    coordinates: tuple[float, ...] = RETREAT_COORDINATES,
    resolution: float = RETREAT_RESOLUTION,
) -> RetreatEvidence | None:
    """Retreat along the snapshot-anchored trajectory and return solved feasible evidence.

    Returns None when the snapshot admits no valid convergent retreat under this policy;
    callers own the fallback, because a corpus built on another trajectory legitimately
    fails to produce evidence here.
    """
    try:
        active_policy = build_active_policy(snapshot_net)
    except ValueError:
        return None

    def state_builder(lambda_value: float) -> Any:
        return build_diagnostic_state(
            snapshot_net,
            lambda_value=lambda_value,
            active_policy=active_policy,
        )

    measured = measure_boundary(
        state_builder,
        coarse_coordinates=coordinates,
        refinement_resolution=resolution,
        feasibility_policy=BoundaryFeasibilityPolicy(),
    )
    if measured.status != "BOUNDARY_FOUND" or measured.highest_solved is None:
        return None

    probe = solve_locked_probe(state_builder(measured.highest_solved).net)
    logical = measured.logical_probe_count + 1
    attempts = measured.solver_attempt_count + probe.solver_attempt_count
    if probe.status != "SOLVED":
        return None

    feasibility = evaluate_solved_feasibility(probe.solved_net)
    if not feasibility.feasible:
        return None

    return RetreatEvidence(
        lambda_value=float(measured.highest_solved),
        solved_net=probe.solved_net,
        q_status=tuple(feasibility.generator_q_status),
        logical_probe_count=logical,
        solver_attempt_count=attempts,
    )
