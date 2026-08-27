# ABOUTME: Constructs reactive-deficit curation families from immutable profiles and pockets.
# ABOUTME: Delegates trajectory, active balance, solver, feasibility, and boundary logic to plan 15.
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from restorebench.physics.active_balance import build_active_policy
from restorebench.physics.boundary import measure_boundary
from restorebench.physics.feasibility import evaluate_solved_feasibility
from restorebench.physics.policies import (
    CURATION_TRAJECTORY_POLICY_VERSION,
)
from restorebench.physics.solver import solve_locked_probe
from restorebench.physics.trajectory import build_curation_state
from restorebench.schemas.dataset import (
    ActiveScheduleRecipe,
    BaseGeneratorDispatch,
    BoundaryInterval,
    CurationRecipe,
    MonotonicityObservation,
    PocketRecipe,
)
from restorebench.schemas.physics import (
    ActiveBalancePolicy,
    BoundaryFeasibilityPolicy,
    CurationLoadWeight,
    TrajectoryState,
)
from restorebench.corpus.operating_profiles import OperatingProfileCandidate


CURATION_SCAN_POLICY_VERSION = "reactive-deficit-coarse-scan-v1"
TARGET_DEPTH_POLICY_VERSION = "boundary-relative-target-depth-v1"


class CurationScanPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    coarse_coordinates: tuple[float, ...]
    refinement_resolution: float = Field(gt=0.0, allow_inf_nan=False)
    # Voltage-only violations remain useful for measuring solver behavior, but
    # can never become admission evidence. Other electrical infeasibility still
    # terminates the family inside the shared boundary primitive.
    stop_on_solved_infeasibility: bool = False
    policy_version: str = CURATION_SCAN_POLICY_VERSION

    @model_validator(mode="after")
    def coordinates_are_valid(self) -> "CurationScanPolicy":
        if len(self.coarse_coordinates) < 2:
            raise ValueError("curation scan requires at least two coarse coordinates")
        if not all(
            math.isfinite(coordinate) and coordinate >= 0.0
            for coordinate in self.coarse_coordinates
        ):
            raise ValueError("curation coordinates must be finite and non-negative")
        if any(
            right <= left
            for left, right in zip(
                self.coarse_coordinates,
                self.coarse_coordinates[1:],
                strict=False,
            )
        ):
            raise ValueError("curation coordinates must be strictly increasing")
        if self.coarse_coordinates[0] != 0.0:
            raise ValueError("curation scan must begin at stress zero")
        return self


class CurationFamilyMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    scenario_family_id: str
    profile_id: str
    pocket: PocketRecipe
    active_policy: ActiveBalancePolicy
    boundary: BoundaryInterval
    monotonicity: MonotonicityObservation
    base_solution: Any
    last_convergent_state: TrajectoryState
    last_convergent_solution: Any
    logical_probe_count: int
    solver_attempt_count: int
    scan_policy_version: str


def measure_curation_family(
    profile: OperatingProfileCandidate,
    pocket: PocketRecipe,
    *,
    scan_policy: CurationScanPolicy,
) -> CurationFamilyMeasurement:
    """Measure one scan-supported family boundary and retain valid solved evidence."""
    active_policy = build_active_policy(profile.net)
    weights = _trajectory_weights(pocket)

    base_state = build_curation_state(
        profile.net,
        stress=0.0,
        ordered_load_weights=weights,
        active_policy=active_policy,
    )
    base_probe = solve_locked_probe(base_state.net)
    if base_probe.status != "SOLVED":
        raise ValueError("operating profile does not solve at curation stress zero")
    base_feasibility = evaluate_solved_feasibility(base_probe.solved_net)
    if not base_feasibility.feasible:
        raise ValueError("operating profile is infeasible at curation stress zero")

    def state_builder(stress: float) -> TrajectoryState:
        return build_curation_state(
            profile.net,
            stress=stress,
            ordered_load_weights=weights,
            active_policy=active_policy,
        )

    measured = measure_boundary(
        state_builder,
        coarse_coordinates=scan_policy.coarse_coordinates,
        refinement_resolution=scan_policy.refinement_resolution,
        feasibility_policy=BoundaryFeasibilityPolicy(
            stop_on_solved_infeasibility=scan_policy.stop_on_solved_infeasibility,
        ),
    )
    if measured.status != "BOUNDARY_FOUND":
        raise ValueError(f"curation family rejected with boundary status {measured.status}")
    if measured.highest_solved is None or measured.lowest_unsolved is None:
        raise AssertionError("boundary result omitted its solved/unsolved interval")

    valid_evidence_records = [
        record
        for record in measured.records
        if (
            record.logical_result == "SOLVED"
            and record.feasible is True
            and record.coordinate <= measured.highest_solved
        )
    ]
    if not valid_evidence_records:
        raise ValueError(
            "boundary has no electrically valid constrained evidence state"
        )
    evidence_coordinate = max(
        record.coordinate
        for record in valid_evidence_records
    )
    evidence_state = state_builder(evidence_coordinate)
    evidence_probe = solve_locked_probe(evidence_state.net)
    if evidence_probe.status != "SOLVED":
        raise ValueError("recorded highest-solved boundary point did not reproduce")
    evidence_feasibility = evaluate_solved_feasibility(evidence_probe.solved_net)
    if not evidence_feasibility.feasible:
        raise ValueError("last convergent curation evidence is infeasible")

    coarse_records = sorted(
        (
            record
            for record in measured.records
            if record.phase == "COARSE"
        ),
        key=lambda record: record.coordinate,
    )
    monotonicity = MonotonicityObservation(
        status="OBSERVED_MONOTONIC",
        probe_coordinates=tuple(record.coordinate for record in coarse_records),
        probe_statuses=tuple(
            "SOLVED"
            if record.logical_result == "SOLVED"
            else record.probe_status
            for record in coarse_records
        ),
        policy_version=scan_policy.policy_version,
    )
    return CurationFamilyMeasurement(
        scenario_family_id=scenario_family_id(
            profile_id=profile.profile_id,
            pocket=pocket,
            active_policy=active_policy,
        ),
        profile_id=profile.profile_id,
        pocket=pocket,
        active_policy=active_policy,
        boundary=BoundaryInterval(
            lower=measured.highest_solved,
            upper=measured.lowest_unsolved,
            resolution=scan_policy.refinement_resolution,
            capped=False,
        ),
        monotonicity=monotonicity,
        base_solution=base_probe.solved_net,
        last_convergent_state=evidence_state,
        last_convergent_solution=evidence_probe.solved_net,
        logical_probe_count=measured.logical_probe_count + 2,
        solver_attempt_count=(
            measured.solver_attempt_count
            + base_probe.solver_attempt_count
            + evidence_probe.solver_attempt_count
        ),
        scan_policy_version=scan_policy.policy_version,
    )


def build_target_state(
    profile_net: Any,
    *,
    pocket: PocketRecipe,
    target_stress: float,
    active_policy: ActiveBalancePolicy,
) -> TrajectoryState:
    """Reconstruct one target directly from the immutable profile arrays."""
    state = build_curation_state(
        profile_net,
        stress=target_stress,
        ordered_load_weights=_trajectory_weights(pocket),
        active_policy=active_policy,
    )
    if state.active_balance.status == "ACTIVE_HEADROOM_EXHAUSTED":
        raise ValueError("target reaches active headroom before Q-V admission")
    return state


def build_curation_recipe(
    profile: OperatingProfileCandidate,
    *,
    pocket: PocketRecipe,
    target_stress: float,
    active_policy: ActiveBalancePolicy,
) -> CurationRecipe:
    """Serialize immutable profile, pocket, and active-schedule inputs."""
    active_schedule = ActiveScheduleRecipe(
        generators=tuple(
            BaseGeneratorDispatch(
                gen_id=item.gen_id,
                base_p_mw=float(profile.net.gen.at[item.gen_id, "p_mw"]),
            )
            for item in active_policy.participation
        ),
        participation_factors=tuple(
            item.factor
            for item in active_policy.participation
        ),
        policy_version=active_policy.policy_version,
    )
    base_payload = {
        "operating_profile_id": profile.profile_id,
        "base_state_hash": profile.state_hash,
        "pocket": pocket.model_dump(mode="json"),
        "active_schedule": active_schedule.model_dump(mode="json"),
        "target_stress": target_stress,
    }
    recipe_hash = _hash_payload(base_payload)
    return CurationRecipe(
        **base_payload,
        recipe_hash=recipe_hash,
    )


def scenario_family_id(
    *,
    profile_id: str,
    pocket: PocketRecipe,
    active_policy: ActiveBalancePolicy,
) -> str:
    """Hash family identity before any target-depth choice."""
    payload = {
        "operating_profile_id": profile_id,
        "pocket_anchor": pocket.anchor_bus,
        "ordered_load_weights": [
            (point.load_id, point.weight)
            for point in pocket.loads
        ],
        "pocket_policy_version": pocket.policy_version,
        "load_policy_version": CURATION_TRAJECTORY_POLICY_VERSION,
        "active_policy_version": active_policy.policy_version,
    }
    return f"F-{_hash_payload(payload)[:20]}"


def _trajectory_weights(
    pocket: PocketRecipe,
) -> tuple[CurationLoadWeight, ...]:
    return tuple(
        CurationLoadWeight(load_id=point.load_id, weight=point.weight)
        for point in pocket.loads
    )


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
