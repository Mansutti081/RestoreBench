# ABOUTME: Implements target-side reactive-deficit admission evidence on fresh network copies.
# ABOUTME: Separates locked failure, alternative initialization, Q-unlimited, and valid-state Q-V evidence.
from __future__ import annotations

import copy
import math
from typing import Any

import pandapower as pp
from pandapower.auxiliary import LoadflowNotConverged
from pydantic import BaseModel, ConfigDict, Field

from restorebench.physics.electrical_distance import impedance_weighted_graph_distances
from restorebench.physics.feasibility import evaluate_solved_feasibility
from restorebench.physics.fingerprint import state_fingerprint
from restorebench.physics.solver import (
    ALGORITHM,
    CHECK_CONNECTIVITY,
    MAX_ITERATION,
    PRIMARY_TOLERANCE_MVA,
    RECOVERY_TOLERANCE_MVA,
    solve_locked_probe,
)
from restorebench.schemas.dataset import (
    AlternativeInitializationAudit,
    CurationRecipe,
    PocketRecipe,
    QUnlimitedCounterfactual,
    QVEvidence,
)
from restorebench.schemas.physics import ElectricalDistancePolicy
from restorebench.corpus.curation import build_curation_recipe, build_target_state


ALTERNATIVE_INIT_POLICY_VERSION = "flat-fresh-copy-v1"
Q_UNLIMITED_POLICY_VERSION = "q-unlimited-counterfactual-v1"
QV_THRESHOLDS_POLICY_VERSION = "reactive-deficit-evidence-thresholds-v2"


class ReactiveDeficitThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_voltage_deterioration_pu: float = Field(
        gt=0.0,
        allow_inf_nan=False,
    )
    weak_bus_band_pu: float = Field(gt=0.0, allow_inf_nan=False)
    minimum_q_headroom_reduction_mvar: float = Field(
        gt=0.0,
        allow_inf_nan=False,
    )
    material_q_violation_floor_mvar: float = Field(
        gt=0.0,
        allow_inf_nan=False,
    )
    material_q_violation_fraction: float = Field(
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    maximum_weak_region_distance_pu: float = Field(
        gt=0.0,
        allow_inf_nan=False,
    )
    common_mva_base: float = Field(default=100.0, gt=0.0, allow_inf_nan=False)
    minimum_edge_weight_pu: float = Field(
        default=1e-6,
        gt=0.0,
        allow_inf_nan=False,
    )
    policy_version: str = QV_THRESHOLDS_POLICY_VERSION


class TargetAdmissionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    target_state: Any
    recipe: CurationRecipe
    target_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    alternative_init_audit: AlternativeInitializationAudit
    q_unlimited_counterfactual: QUnlimitedCounterfactual
    qv_evidence: QVEvidence
    locked_logical_probe_count: int = Field(ge=1)
    locked_solver_attempt_count: int = Field(ge=2)


def admit_target_candidate(
    profile: Any,
    family: Any,
    *,
    target_stress: float,
    thresholds: ReactiveDeficitThresholds,
) -> TargetAdmissionResult:
    """Apply every target-side reactive-deficit gate on fresh reconstruction."""
    target_state = build_target_state(
        profile.net,
        pocket=family.pocket,
        target_stress=target_stress,
        active_policy=family.active_policy,
    )
    if target_state.active_balance.status != "SCHEDULED":
        raise ValueError("target exhausts active-power headroom")
    target_state = _normalize_target_state_for_storage(target_state)

    reconstructed = build_target_state(
        profile.net,
        pocket=family.pocket,
        target_stress=target_stress,
        active_policy=family.active_policy,
    )
    reconstructed = _normalize_target_state_for_storage(reconstructed)
    target_hash = state_fingerprint(target_state.net).value
    if state_fingerprint(reconstructed.net).value != target_hash:
        raise ValueError("target reconstruction is not byte-stable after normalization")

    locked = solve_locked_probe(target_state.net)
    if (
        locked.status != "NO_SOLUTION"
        or locked.solver_attempt_count != 2
        or tuple(attempt.status for attempt in locked.attempts)
        != ("NO_SOLUTION", "NO_SOLUTION")
    ):
        raise ValueError(
            "locked target must fail primary and recovery attempts"
        )

    alternative = audit_alternative_initialization(target_state.net)
    if alternative.converged_without_action:
        raise ValueError(
            "target converges under alternative initialization without action"
        )
    counterfactual = evaluate_q_unlimited_counterfactual(
        target_state.net,
        thresholds=thresholds,
    )
    qv_evidence = build_qv_evidence(
        base_solution=family.base_solution,
        evidence_solution=family.last_convergent_solution,
        evidence_stress=family.last_convergent_state.coordinate,
        pocket=family.pocket,
        thresholds=thresholds,
    )
    recipe = build_curation_recipe(
        profile,
        pocket=family.pocket,
        target_stress=target_stress,
        active_policy=family.active_policy,
    )
    normalized = CurationRecipe.model_validate_json(
        recipe.model_dump_json()
    )
    if normalized != recipe:
        raise ValueError("curation recipe is not stable after normalized serialization")

    return TargetAdmissionResult(
        target_state=target_state,
        recipe=recipe,
        target_fingerprint=target_hash,
        alternative_init_audit=alternative,
        q_unlimited_counterfactual=counterfactual,
        qv_evidence=qv_evidence,
        locked_logical_probe_count=locked.logical_probe_count,
        locked_solver_attempt_count=locked.solver_attempt_count,
    )


def _normalize_target_state_for_storage(target_state: Any) -> Any:
    """Round-trip the target before any gate so curation matches the public snapshot."""
    normalized_state = copy.deepcopy(target_state)
    storage_input = copy.deepcopy(target_state.net)
    pp.reset_results(storage_input)
    serialized = pp.to_json(storage_input)
    if not isinstance(serialized, str):
        raise TypeError("pandapower in-memory serialization did not return JSON")
    normalized_net = pp.from_json(serialized)
    normalized_state.net = normalized_net
    normalized_state.active_balance.net = normalized_net
    return normalized_state


def audit_alternative_initialization(
    net: Any,
    *,
    init_policy: str = "flat",
) -> AlternativeInitializationAudit:
    """Run the two locked tolerances on fresh copies while changing only initialization."""
    statuses: list[str] = []
    for tolerance in (PRIMARY_TOLERANCE_MVA, RECOVERY_TOLERANCE_MVA):
        attempt = copy.deepcopy(net)
        try:
            pp.runpp(
                attempt,
                algorithm=ALGORITHM,
                enforce_q_lims=True,
                init=init_policy,
                tolerance_mva=tolerance,
                max_iteration=MAX_ITERATION,
                check_connectivity=CHECK_CONNECTIVITY,
            )
        except LoadflowNotConverged:
            statuses.append("NO_SOLUTION")
            continue
        if bool(getattr(attempt, "converged", False)):
            statuses.append("SOLVED")
            break
        statuses.append("NO_SOLUTION")

    primary_status = statuses[0]
    if primary_status == "SOLVED":
        recovery_status = "NOT_RUN"
    else:
        recovery_status = statuses[1]
    converged = "SOLVED" in statuses
    return AlternativeInitializationAudit(
        init_policy=init_policy,
        primary_status=primary_status,
        recovery_status=recovery_status,
        converged_without_action=converged,
        solver_attempt_count=len(statuses),
    )


def evaluate_q_unlimited_counterfactual(
    net: Any,
    *,
    thresholds: ReactiveDeficitThresholds,
) -> QUnlimitedCounterfactual:
    """Require a valid Q-unlimited solution with material declared-Q violations."""
    solved = _solve_q_unlimited(net)
    violation_ids: list[int] = []
    max_violation = 0.0
    for gen_id, row in solved.gen.sort_index().iterrows():
        if not bool(row.get("in_service", True)):
            continue
        if gen_id not in solved.res_gen.index:
            raise ValueError("Q-unlimited solution is missing generator results")
        q_mvar = float(solved.res_gen.at[gen_id, "q_mvar"])
        min_q = float(row["min_q_mvar"])
        max_q = float(row["max_q_mvar"])
        if not all(math.isfinite(value) for value in (q_mvar, min_q, max_q)):
            raise ValueError("Q-unlimited generator evidence must be finite")
        violation = max(q_mvar - max_q, min_q - q_mvar, 0.0)
        material_threshold = max(
            thresholds.material_q_violation_floor_mvar,
            thresholds.material_q_violation_fraction * abs(max_q - min_q),
        )
        if violation >= material_threshold:
            violation_ids.append(int(gen_id))
            max_violation = max(max_violation, violation)
    if not violation_ids:
        raise ValueError(
            "Q-unlimited counterfactual has no material generator-Q violation"
        )
    structural = evaluate_solved_feasibility(solved)
    if not (
        structural.generator_p_within_limits
        and structural.connected
        and structural.loads_energized
    ):
        raise ValueError(
            "Q-unlimited counterfactual violates active or structural feasibility"
        )
    if not _external_grid_within_limits(solved):
        raise ValueError("Q-unlimited counterfactual violates external-grid limits")
    return QUnlimitedCounterfactual(
        converged=True,
        material_violation_gen_ids=tuple(violation_ids),
        max_violation_mvar=max_violation,
        ext_grid_feasible=True,
        policy_version=Q_UNLIMITED_POLICY_VERSION,
    )


def build_qv_evidence(
    *,
    base_solution: Any,
    evidence_solution: Any,
    evidence_stress: float,
    pocket: PocketRecipe,
    thresholds: ReactiveDeficitThresholds,
) -> QVEvidence:
    """Compare valid constrained solutions and record weak-region locality."""
    base = evaluate_solved_feasibility(base_solution)
    evidence = evaluate_solved_feasibility(evidence_solution)
    if not base.feasible or not evidence.feasible:
        raise ValueError("Q-V evidence states must both be electrically feasible")

    base_min_vm = base.voltage.min_vm_pu
    evidence_min_vm = evidence.voltage.min_vm_pu
    voltage_deterioration = base_min_vm - evidence_min_vm
    if voltage_deterioration < thresholds.minimum_voltage_deterioration_pu:
        raise ValueError(
            "insufficient voltage deterioration for reactive-deficit admission"
        )
    voltages = evidence_solution.res_bus["vm_pu"].dropna().astype(float)
    weak_ids = tuple(
        int(bus_id)
        for bus_id in sorted(
            voltages.index[
                voltages <= evidence_min_vm + thresholds.weak_bus_band_pu
            ]
        )
    )
    if not weak_ids:
        raise ValueError("Q-V evidence produced no weak buses")
    distances = impedance_weighted_graph_distances(
        evidence_solution,
        source_buses=(pocket.anchor_bus,),
        policy=ElectricalDistancePolicy(
            common_mva_base=thresholds.common_mva_base,
            minimum_edge_weight_pu=thresholds.minimum_edge_weight_pu,
        ),
    )
    weak_distances = tuple(
        float(distance)
        for bus_id in weak_ids
        if (distance := distances.distances_pu.get(bus_id)) is not None
    )
    if not weak_distances:
        raise ValueError(
            "weak Q-V region distance from the stressed pocket is unavailable"
        )
    weak_region_min_distance = min(weak_distances)
    weak_region_local = (
        weak_region_min_distance
        <= thresholds.maximum_weak_region_distance_pu
    )

    base_by_id = {item.gen_id: item for item in base.generator_q_status}
    evidence_by_id = {
        item.gen_id: item
        for item in evidence.generator_q_status
    }
    common_ids = sorted(set(base_by_id) & set(evidence_by_id))
    base_upper_headroom = math.fsum(
        max(float(base_by_id[gen_id].upper_headroom_mvar or 0.0), 0.0)
        for gen_id in common_ids
    )
    evidence_upper_headroom = math.fsum(
        max(float(evidence_by_id[gen_id].upper_headroom_mvar or 0.0), 0.0)
        for gen_id in common_ids
    )
    reduction = base_upper_headroom - evidence_upper_headroom
    if reduction < thresholds.minimum_q_headroom_reduction_mvar:
        raise ValueError(
            "insufficient upper-Q headroom reduction for reactive-deficit admission"
        )

    q_limited_ids = tuple(
        item.gen_id
        for item in evidence.generator_q_status
        if item.status in {"Q_LIMITED_UPPER", "Q_LIMITED_LOWER"}
    )
    newly_limited_ids = tuple(
        gen_id
        for gen_id in q_limited_ids
        if base_by_id[gen_id].status
        not in {"Q_LIMITED_UPPER", "Q_LIMITED_LOWER"}
    )
    upper_near_ids = tuple(
        item.gen_id
        for item in evidence.generator_q_status
        if item.status == "Q_LIMITED_UPPER"
        or (
            item.status == "PV_CONTROLLABLE"
            and item.upper_headroom_mvar is not None
            and item.upper_headroom_mvar
            < max(
                thresholds.material_q_violation_floor_mvar,
                thresholds.material_q_violation_fraction
                * abs(item.max_q_mvar - item.min_q_mvar),
            )
        )
    )
    if not upper_near_ids:
        raise ValueError("Q-V evidence has no upper-Q-limited or near-limited generator")
    slack = evidence.slack_results[0]
    return QVEvidence(
        evidence_stress=evidence_stress,
        weak_bus_ids=weak_ids,
        weak_region_min_distance_pu=weak_region_min_distance,
        weak_region_local=weak_region_local,
        min_vm_pu=evidence_min_vm,
        q_near_limit_gen_ids=upper_near_ids,
        q_limited_gen_ids=q_limited_ids,
        newly_q_limited_gen_ids=newly_limited_ids,
        q_headroom_reduction_mvar=reduction,
        generator_p_feasible=True,
        ext_grid_p_mw=slack.p_mw,
        ext_grid_q_mvar=slack.q_mvar,
        hard_voltage_envelope_passed=True,
        thresholds_version=thresholds.policy_version,
    )


def _solve_q_unlimited(net: Any) -> Any:
    last_error: Exception | None = None
    for tolerance in (PRIMARY_TOLERANCE_MVA, RECOVERY_TOLERANCE_MVA):
        attempt = copy.deepcopy(net)
        try:
            pp.runpp(
                attempt,
                algorithm=ALGORITHM,
                enforce_q_lims=False,
                init="dc",
                tolerance_mva=tolerance,
                max_iteration=MAX_ITERATION,
                check_connectivity=CHECK_CONNECTIVITY,
            )
        except LoadflowNotConverged as exc:
            last_error = exc
            continue
        if bool(getattr(attempt, "converged", False)):
            return attempt
    raise ValueError(
        f"Q-unlimited counterfactual did not converge: {last_error}"
    )


def _external_grid_within_limits(net: Any) -> bool:
    for ext_grid_id, row in net.ext_grid.sort_index().iterrows():
        if not bool(row.get("in_service", True)):
            continue
        if ext_grid_id not in net.res_ext_grid.index:
            return False
        for value_column, lower_column, upper_column in (
            ("p_mw", "min_p_mw", "max_p_mw"),
            ("q_mvar", "min_q_mvar", "max_q_mvar"),
        ):
            value = float(net.res_ext_grid.at[ext_grid_id, value_column])
            lower = float(row.get(lower_column, float("nan")))
            upper = float(row.get(upper_column, float("nan")))
            if not math.isfinite(value):
                return False
            if math.isfinite(lower) and value < lower - 1e-3:
                return False
            if math.isfinite(upper) and value > upper + 1e-3:
                return False
    return True
