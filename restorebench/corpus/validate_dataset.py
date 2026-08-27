# ABOUTME: Independently reconstructs and validates staged reactive-deficit corpus artifacts.
# ABOUTME: Shares only canonical schemas and plan-15 physics, never dataset generator helpers.
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import pandapower as pp
from pandapower.auxiliary import LoadflowNotConverged
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from restorebench.environment.card_render import render_scenario_card
from restorebench.physics.actions import (
    apply_qv_action,
    enumerate_legal_qv_actions,
    get_qv_action_applicability,
)
from restorebench.physics.boundary import measure_boundary
from restorebench.physics.electrical_distance import (
    impedance_weighted_graph_distances,
)
from restorebench.physics.feasibility import (
    evaluate_solved_feasibility,
    satisfies_non_voltage_constraints,
)
from restorebench.physics.fingerprint import state_fingerprint
from restorebench.physics.solver import (
    ALGORITHM,
    CHECK_CONNECTIVITY,
    MAX_ITERATION,
    PRIMARY_TOLERANCE_MVA,
    RECOVERY_TOLERANCE_MVA,
    solve_locked_probe,
)
from restorebench.physics.trajectory import (
    build_curation_state,
    build_diagnostic_state,
)
from restorebench.schemas.dataset import (
    DatasetManifest,
    EvaluationManifest,
    ScenarioLabel,
)
from restorebench.schemas.physics import (
    ActiveBalancePolicy,
    BoundaryFeasibilityPolicy,
    CurationLoadWeight,
    ElectricalDistancePolicy,
    GeneratorParticipation,
)
from restorebench.corpus.checkpoint_io import (
    CheckpointCompatibilityError,
    CheckpointStore,
    atomic_write_json,
)
from restorebench.corpus.versions import VALIDATOR_VERSION


CHECKS = {
    1: "schema_and_shape",
    2: "augmentation_and_profile",
    3: "pocket_and_constant_pf",
    4: "bounded_active_schedule",
    5: "boundary_reconstruction",
    6: "observed_monotonicity",
    7: "locked_target_failure",
    8: "alternative_initialization",
    9: "q_unlimited_counterfactual",
    10: "constrained_qv_evidence",
    11: "hard_voltage_envelope",
    12: "witness_and_descriptors",
    13: "artifact_agreement",
    14: "private_field_denial",
    15: "leakage_split_and_hashes",
}
PRIVATE_TOKENS = (
    "scenario_family_id",
    "leakage_group_id",
    "target_stress",
    "qv_evidence",
    "q_unlimited_counterfactual",
    "alternative_init_audit",
    "resolution_regime",
    "witness_length",
    "target_depth",
    "pre_weakening",
    "cause_hint",
    "lambda_nose",
)
LEAN_COLUMNS = {
    "bus": {"vn_kv", "in_service"},
    "line": {"from_bus", "to_bus", "in_service"},
    "trafo": {
        "hv_bus",
        "lv_bus",
        "tap_side",
        "tap_pos",
        "tap_min",
        "tap_max",
        "tap_step_percent",
        "in_service",
    },
    "shunt": {
        "bus",
        "p_mw",
        "q_mvar",
        "step",
        "max_step",
        "in_service",
    },
    "gen": {
        "bus",
        "vm_pu",
        "p_mw",
        "min_p_mw",
        "max_p_mw",
        "min_q_mvar",
        "max_q_mvar",
        "in_service",
    },
    "load": {"bus", "p_mw", "q_mvar", "in_service"},
    "ext_grid": {
        "bus",
        "vm_pu",
        "min_p_mw",
        "max_p_mw",
        "min_q_mvar",
        "max_q_mvar",
        "in_service",
    },
}


class ScenarioValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    valid: bool
    failed_checks: list[str] = Field(default_factory=list)
    details: dict[str, list[str]] = Field(default_factory=dict)


class CorpusValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    validator_version: str
    dataset_dir: str
    valid: bool
    total: int
    valid_count: int
    invalid_count: int
    failed_check_counts: dict[str, int]
    reports: list[ScenarioValidation]


def validate_corpus(
    dataset_dir: str | Path,
    *,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> CorpusValidation:
    root = Path(dataset_dir)
    if resume and checkpoint_dir is None:
        raise ValueError("validation resume requires an explicit checkpoint directory")
    manifest = DatasetManifest.model_validate_json(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    evaluation_path = root / "evaluation_manifest.json"
    evaluation = EvaluationManifest.model_validate_json(
        evaluation_path.read_text(encoding="utf-8")
    )
    generation_report = _load_generation_report(root)
    raw_policy = _generation_policy(generation_report)
    declared_split_counts = _declared_split_counts(
        generation_report,
        scenario_count=manifest.scenario_count,
    )
    labels = _load_models(
        root / "private/labels.json",
        ScenarioLabel,
    )
    witnesses = _load_raw_list(root / "private/witnesses.json")
    labels_by_id = {label.scenario_id: label for label in labels}
    witnesses_by_id = {
        str(witness.get("scenario_id")): witness
        for witness in witnesses
    }
    reports: list[ScenarioValidation] = []

    global_failures = _global_contract_failures(
        root=root,
        manifest=manifest,
        evaluation=evaluation,
        labels=labels,
        witnesses=witnesses,
        raw_policy=raw_policy,
        declared_split_counts=declared_split_counts,
    )
    checkpoint_store: CheckpointStore | None = None
    if checkpoint_dir is not None:
        _require_safe_validation_checkpoint(checkpoint_dir)
        checkpoint_store = CheckpointStore.open(
            checkpoint_dir,
            identity=_validation_checkpoint_identity(root),
            resume=resume,
        )
    for entry in manifest.scenarios:
        reports.append(
            _validate_scenario_with_checkpoint(
                root=root,
                entry=entry,
                label=labels_by_id.get(entry.scenario_id),
                witness=witnesses_by_id.get(entry.scenario_id),
                raw_policy=raw_policy,
                global_messages=global_failures.get(
                    entry.scenario_id,
                    (),
                ),
                checkpoint_store=checkpoint_store,
            )
        )

    valid_count = sum(report.valid for report in reports)
    failed_counts = Counter(
        check
        for report in reports
        for check in report.failed_checks
    )
    return CorpusValidation(
        validator_version=VALIDATOR_VERSION,
        dataset_dir=str(root.resolve()),
        valid=(
            bool(reports)
            and len(reports) == manifest.scenario_count
            and valid_count == len(reports)
        ),
        total=len(reports),
        valid_count=valid_count,
        invalid_count=len(reports) - valid_count,
        failed_check_counts=dict(sorted(failed_counts.items())),
        reports=reports,
    )


def _validate_scenario_with_checkpoint(
    *,
    root: Path,
    entry: Any,
    label: ScenarioLabel | None,
    witness: dict[str, Any] | None,
    raw_policy: dict[str, Any],
    global_messages: Sequence[str],
    checkpoint_store: CheckpointStore | None,
) -> ScenarioValidation:
    scenario_id = str(entry.scenario_id)
    if checkpoint_store is not None:
        stored = checkpoint_store.read_json_shard(scenario_id)
        if stored is not None:
            report = ScenarioValidation.model_validate(stored)
            if report.scenario_id != scenario_id:
                raise CheckpointCompatibilityError(
                    f"validation checkpoint scenario ID differs for {scenario_id}"
                )
            print(
                f"VALIDATION_RESUME scenario={scenario_id}",
                flush=True,
            )
            return report

    report = ScenarioValidation(
        scenario_id=scenario_id,
        valid=True,
    )
    for message in global_messages:
        _fail(report, 15, message)
    if label is None or witness is None:
        _fail(report, 1, "private label or witness is missing")
    else:
        try:
            _validate_scenario(
                root=root,
                entry=entry,
                label=label,
                witness=witness,
                raw_policy=raw_policy,
                report=report,
            )
        except Exception as exc:
            _fail(
                report,
                1,
                f"unexpected independent-validation error: {exc}",
            )
    finished = _finish(report)
    if checkpoint_store is not None:
        checkpoint_store.write_json_shard(
            scenario_id,
            finished.model_dump(mode="json"),
        )
        print(
            f"VALIDATION_CHECKPOINT scenario={scenario_id} "
            f"valid={finished.valid}",
            flush=True,
        )
    return finished


def _validate_scenario(
    *,
    root: Path,
    entry: Any,
    label: ScenarioLabel,
    witness: dict[str, Any],
    raw_policy: dict[str, Any],
    report: ScenarioValidation,
) -> None:
    full_path = root / "full" / f"{label.scenario_id}.json"
    lean_path = root / "lean" / f"{label.scenario_id}.json"
    card_path = root / "llm" / f"{label.scenario_id}.md"
    if not all(path.is_file() for path in (full_path, lean_path, card_path)):
        _fail(report, 13, "one or more public artifacts are missing")
        return
    full = pp.from_json(str(full_path))
    network_id = str(raw_policy.get("network_id", "case118"))

    _check_shape_and_results(full, report, network_id=network_id)
    profile = _reconstruct_profile(label, raw_policy, report, network_id=network_id)
    if profile is None:
        return
    pocket_scale = _verify_pocket(label, profile, raw_policy, report)
    if pocket_scale is None:
        return
    active_policy = _active_policy(label, profile, report)
    if active_policy is None:
        return

    weights = tuple(
        CurationLoadWeight(
            load_id=point.load_id,
            weight=point.weight,
        )
        for point in label.recipe.pocket.loads
    )

    def state_builder(stress: float) -> Any:
        return build_curation_state(
            profile,
            stress=stress,
            ordered_load_weights=weights,
            active_policy=active_policy,
        )

    target = state_builder(label.recipe.target_stress)
    if target.active_balance.status != "SCHEDULED":
        _fail(report, 4, "target active schedule is exhausted")
    if not _storage_equivalent(target.net, full):
        _fail(report, 3, "FULL snapshot differs from recipe reconstruction")
    _verify_active_schedule(label, target, report)
    _verify_recipe_hash(label, report)

    scan = raw_policy["scan"]
    measured = measure_boundary(
        state_builder,
        coarse_coordinates=tuple(scan["coarse_coordinates"]),
        refinement_resolution=float(scan["refinement_resolution"]),
        feasibility_policy=BoundaryFeasibilityPolicy(
            stop_on_solved_infeasibility=bool(
                scan["stop_on_solved_infeasibility"]
            )
        ),
    )
    if (
        measured.status != "BOUNDARY_FOUND"
        or not _close(
            measured.highest_solved,
            label.convergence_boundary.lower,
        )
        or not _close(
            measured.lowest_unsolved,
            label.convergence_boundary.upper,
        )
        or not _close(
            label.convergence_boundary.resolution,
            scan["refinement_resolution"],
        )
        or label.convergence_boundary.capped
    ):
        _fail(report, 5, "independent boundary interval differs")
    _verify_monotonicity(label, measured, report)

    locked = solve_locked_probe(full)
    if (
        locked.status != "NO_SOLUTION"
        or tuple(attempt.status for attempt in locked.attempts)
        != ("NO_SOLUTION", "NO_SOLUTION")
    ):
        _fail(report, 7, "target does not fail both locked attempts")
    alternative = _alternative_audit(full)
    expected_alternative = label.alternative_init_audit
    if alternative != {
        "init_policy": expected_alternative.init_policy,
        "primary_status": expected_alternative.primary_status,
        "recovery_status": expected_alternative.recovery_status,
        "converged_without_action": (
            expected_alternative.converged_without_action
        ),
        "solver_attempt_count": expected_alternative.solver_attempt_count,
    }:
        _fail(report, 8, "alternative-initialization audit differs")

    unlimited = _solve_unlimited(full)
    _verify_unlimited(label, unlimited, raw_policy, report)

    evidence_state = state_builder(label.qv_evidence.evidence_stress)
    evidence_probe = solve_locked_probe(evidence_state.net)
    base_probe = solve_locked_probe(profile)
    if evidence_probe.status != "SOLVED" or base_probe.status != "SOLVED":
        _fail(report, 10, "base or Q-V evidence does not converge")
    else:
        _verify_qv_evidence(
            label,
            base_probe.solved_net,
            evidence_probe.solved_net,
            raw_policy,
            report,
        )
        for name, solved in (
            ("base", base_probe.solved_net),
            ("evidence", evidence_probe.solved_net),
        ):
            if not evaluate_solved_feasibility(solved).feasible:
                _fail(report, 11, f"{name} fails the hard envelope")

    _verify_witness(
        label,
        witness,
        full,
        raw_policy,
        report,
    )
    _verify_artifacts(
        entry,
        full,
        full_path,
        lean_path,
        card_path,
        report,
    )
    _verify_private_denial(
        full_path,
        lean_path,
        card_path,
        report,
    )


def _independent_augmented_base(network_id: str = "case118") -> Any:
    if network_id == "case118":
        net = pp.networks.case118()
        frozen = net.gen["p_mw"].abs() < 1e-9
    elif network_id == "case89pegase":
        net = pp.networks.case89pegase()
        frozen = net.gen["p_mw"] <= 1e-9
        positive = net.gen["p_mw"] > 1e-9
        net.gen.loc[positive, "max_p_mw"] *= 1.5
        net.ext_grid.loc[:, "max_q_mvar"] = 2500.0
    else:
        raise ValueError(f"unsupported dataset network: {network_id}")
    tappable = net.trafo["tap_pos"].notna()
    net.trafo.loc[tappable, "tap_min"] = -2
    net.trafo.loc[tappable, "tap_max"] = 2
    net.gen.loc[:, "vm_pu"] = net.gen["vm_pu"].clip(0.95, 1.05)
    net.gen.loc[frozen, "min_p_mw"] = net.gen.loc[frozen, "p_mw"]
    net.gen.loc[frozen, "max_p_mw"] = net.gen.loc[frozen, "p_mw"]
    pp.reset_results(net)
    return net


def _storage_equivalent(reconstructed: Any, stored: Any) -> bool:
    """Compare exact electrical state after pandapower's public JSON normalization."""
    serialized = pp.to_json(reconstructed)
    if not isinstance(serialized, str):
        raise TypeError("pandapower in-memory serialization did not return JSON")
    normalized = pp.from_json(serialized)
    return (
        state_fingerprint(normalized).value
        == state_fingerprint(stored).value
    )


def _reconstruct_profile(
    label: ScenarioLabel,
    raw_policy: dict[str, Any],
    report: ScenarioValidation,
    *,
    network_id: str = "case118",
) -> Any | None:
    net = _independent_augmented_base(network_id)
    profile_policy = raw_policy["operating_profiles"]
    atomic = _independent_profile_modifications(net)
    selected: tuple[dict[str, Any], ...] | None = None
    for size in range(
        int(profile_policy["max_simultaneous_deviations"]) + 1
    ):
        for modifications in combinations(atomic, size):
            keys = {
                (
                    modification["component_type"],
                    modification["component_id"],
                )
                for modification in modifications
            }
            if len(keys) != len(modifications):
                continue
            if (
                _profile_id(
                    modifications,
                    str(profile_policy["policy_version"]),
                )
                == label.recipe.operating_profile_id
            ):
                selected = modifications
                break
        if selected is not None:
            break
    if selected is None:
        _fail(report, 2, "operating profile ID cannot be regenerated")
        return None
    for modification in selected:
        table = (
            net.shunt
            if modification["component_type"] == "SHUNT"
            else net.trafo
        )
        table.at[
            modification["component_id"],
            modification["field"],
        ] = modification["target_value"]
    pp.reset_results(net)
    expected_hash = state_fingerprint(
        net,
        policy_versions={
            "augmentation": (
                label.generation_metadata.curation_policy_versions.augmentation
            ),
            "operating_profile": str(
                profile_policy["policy_version"]
            ),
        },
    ).value
    if expected_hash != label.recipe.base_state_hash:
        _fail(report, 2, "operating profile state hash differs")
    probe = solve_locked_probe(net)
    if (
        probe.status != "SOLVED"
        or not evaluate_solved_feasibility(probe.solved_net).feasible
    ):
        _fail(report, 2, "operating profile is not independently admissible")
    return net


def _independent_profile_modifications(
    net: Any,
) -> tuple[dict[str, Any], ...]:
    modifications: list[dict[str, Any]] = []
    for shunt_id, row in net.shunt.sort_index().iterrows():
        if not bool(row.get("in_service", True)):
            continue
        step = int(row["step"])
        max_step = int(row["max_step"])
        if max_step == 1 and step in {0, 1}:
            modifications.append(
                {
                    "component_type": "SHUNT",
                    "component_id": int(shunt_id),
                    "field": "step",
                    "base_value": step,
                    "target_value": 1 - step,
                }
            )
    for trafo_id, row in net.trafo.sort_index().iterrows():
        if (
            not bool(row.get("in_service", True))
            or math.isnan(float(row["tap_pos"]))
        ):
            continue
        current = int(row["tap_pos"])
        lower = int(row["tap_min"])
        upper = int(row["tap_max"])
        for target in (current - 1, current + 1):
            if lower <= target <= upper:
                modifications.append(
                    {
                        "component_type": "TRAFO",
                        "component_id": int(trafo_id),
                        "field": "tap_pos",
                        "base_value": current,
                        "target_value": target,
                    }
                )
    return tuple(modifications)


def _profile_id(
    modifications: tuple[dict[str, Any], ...],
    policy_version: str,
) -> str:
    payload = {
        "modifications": list(modifications),
        "policy_version": policy_version,
    }
    return f"OP-{_sha256_payload(payload)[:16]}"


def _verify_pocket(
    label: ScenarioLabel,
    profile: Any,
    raw_policy: dict[str, Any],
    report: ScenarioValidation,
) -> float | None:
    recipe = label.recipe.pocket
    ids = tuple(int(load_id) for load_id in profile.load.index)
    if tuple(point.load_id for point in recipe.loads) != ids:
        _fail(report, 3, "pocket does not cover the immutable load table")
        return None
    for point in recipe.loads:
        if not (
            _close(point.base_p_mw, profile.load.at[point.load_id, "p_mw"])
            and _close(
                point.base_q_mvar,
                profile.load.at[point.load_id, "q_mvar"],
            )
        ):
            _fail(report, 3, f"load {point.load_id} base values differ")

    policy = raw_policy["pockets"]
    distance = impedance_weighted_graph_distances(
        profile,
        source_buses=(recipe.anchor_bus,),
        policy=ElectricalDistancePolicy(
            common_mva_base=float(policy["common_mva_base"]),
            minimum_edge_weight_pu=float(
                policy["minimum_edge_weight_pu"]
            ),
        ),
    )
    for scale in policy["distance_scales_pu"]:
        by_bus: dict[int, float] = {}
        for bus in sorted(set(int(value) for value in profile.load["bus"])):
            value = distance.distances_pu.get(bus)
            raw = (
                math.exp(-float(value) / float(scale))
                if value is not None
                else 0.0
            )
            by_bus[bus] = (
                raw
                if raw >= float(policy["weight_cutoff"])
                else 0.0
            )
        maximum = max(by_bus.values())
        weights = {
            bus: round(value / maximum, 12)
            for bus, value in by_bus.items()
        }
        points = [
            {
                "load_id": int(load_id),
                "base_p_mw": float(row["p_mw"]),
                "base_q_mvar": float(row["q_mvar"]),
                "weight": weights[int(row["bus"])],
            }
            for load_id, row in profile.load.sort_index().iterrows()
        ]
        vector_hash = _sha256_payload(
            {
                "anchor_bus": recipe.anchor_bus,
                "distance_scale_pu": float(scale),
                "loads": points,
                "policy": policy,
            }
        )
        if vector_hash != recipe.vector_hash:
            continue
        if points != [
            point.model_dump(mode="json")
            for point in recipe.loads
        ]:
            _fail(report, 3, "pocket weights differ from regenerated vector")
        return float(scale)
    _fail(report, 3, "pocket vector hash cannot be regenerated")
    return None


def _active_policy(
    label: ScenarioLabel,
    profile: Any,
    report: ScenarioValidation,
) -> ActiveBalancePolicy | None:
    schedule = label.recipe.active_schedule
    if len(schedule.generators) != len(schedule.participation_factors):
        _fail(report, 4, "active schedule dimensions differ")
        return None
    participants = []
    for generator, factor in zip(
        schedule.generators,
        schedule.participation_factors,
        strict=True,
    ):
        if not _close(
            generator.base_p_mw,
            profile.gen.at[generator.gen_id, "p_mw"],
        ):
            _fail(report, 4, f"generator {generator.gen_id} base P differs")
        participants.append(
            GeneratorParticipation(
                gen_id=generator.gen_id,
                factor=factor,
            )
        )
    return ActiveBalancePolicy(
        participation=tuple(participants),
        policy_version=schedule.policy_version,
    )


def _verify_active_schedule(
    label: ScenarioLabel,
    target: Any,
    report: ScenarioValidation,
) -> None:
    balance = target.active_balance
    if not (
        balance.status == "SCHEDULED"
        and abs(balance.unallocated_delta_mw) <= 1e-9
        and _close(
            balance.requested_delta_mw,
            target.requested_load_delta_mw,
        )
    ):
        _fail(report, 4, "active schedule does not exactly balance load")
    for item in balance.generator_dispatch:
        minimum = float(target.net.gen.at[item.gen_id, "min_p_mw"])
        maximum = float(target.net.gen.at[item.gen_id, "max_p_mw"])
        if (
            item.scheduled_p_mw < minimum - 1e-9
            or item.scheduled_p_mw > maximum + 1e-9
        ):
            _fail(report, 4, f"generator {item.gen_id} P is out of bounds")


def _verify_recipe_hash(
    label: ScenarioLabel,
    report: ScenarioValidation,
) -> None:
    payload = label.recipe.model_dump(mode="json")
    actual = payload.pop("recipe_hash")
    if _sha256_payload(payload) != actual:
        _fail(report, 3, "curation recipe hash differs")


def _verify_monotonicity(
    label: ScenarioLabel,
    measured: Any,
    report: ScenarioValidation,
) -> None:
    coarse = sorted(
        (
            record
            for record in measured.records
            if record.phase == "COARSE"
        ),
        key=lambda record: record.coordinate,
    )
    coordinates = tuple(record.coordinate for record in coarse)
    statuses = tuple(
        "SOLVED"
        if record.logical_result == "SOLVED"
        else record.probe_status
        for record in coarse
    )
    if (
        label.monotonicity.status != "OBSERVED_MONOTONIC"
        or coordinates != label.monotonicity.probe_coordinates
        or statuses != label.monotonicity.probe_statuses
    ):
        _fail(report, 6, "monotonicity record differs from coarse scan")


def _alternative_audit(net: Any) -> dict[str, Any]:
    statuses: list[str] = []
    for tolerance in (
        PRIMARY_TOLERANCE_MVA,
        RECOVERY_TOLERANCE_MVA,
    ):
        trial = copy.deepcopy(net)
        try:
            pp.runpp(
                trial,
                algorithm=ALGORITHM,
                enforce_q_lims=True,
                init="flat",
                tolerance_mva=tolerance,
                max_iteration=MAX_ITERATION,
                check_connectivity=CHECK_CONNECTIVITY,
            )
        except LoadflowNotConverged:
            statuses.append("NO_SOLUTION")
            continue
        if bool(trial.converged):
            statuses.append("SOLVED")
            break
        statuses.append("NO_SOLUTION")
    primary = statuses[0]
    recovery = "NOT_RUN" if primary == "SOLVED" else statuses[1]
    return {
        "init_policy": "flat",
        "primary_status": primary,
        "recovery_status": recovery,
        "converged_without_action": "SOLVED" in statuses,
        "solver_attempt_count": len(statuses),
    }


def _solve_unlimited(net: Any) -> Any | None:
    for tolerance in (
        PRIMARY_TOLERANCE_MVA,
        RECOVERY_TOLERANCE_MVA,
    ):
        trial = copy.deepcopy(net)
        try:
            pp.runpp(
                trial,
                algorithm=ALGORITHM,
                enforce_q_lims=False,
                init="dc",
                tolerance_mva=tolerance,
                max_iteration=MAX_ITERATION,
                check_connectivity=CHECK_CONNECTIVITY,
            )
        except LoadflowNotConverged:
            continue
        if bool(trial.converged):
            return trial
    return None


def _verify_unlimited(
    label: ScenarioLabel,
    solved: Any | None,
    raw_policy: dict[str, Any],
    report: ScenarioValidation,
) -> None:
    if solved is None:
        _fail(report, 9, "Q-unlimited target does not converge")
        return
    threshold = raw_policy["reactive_thresholds"]
    violations: list[int] = []
    maximum = 0.0
    for gen_id, row in solved.gen.sort_index().iterrows():
        if not bool(row.get("in_service", True)):
            continue
        q_mvar = float(solved.res_gen.at[gen_id, "q_mvar"])
        lower = float(row["min_q_mvar"])
        upper = float(row["max_q_mvar"])
        violation = max(q_mvar - upper, lower - q_mvar, 0.0)
        material = max(
            float(threshold["material_q_violation_floor_mvar"]),
            float(threshold["material_q_violation_fraction"])
            * abs(upper - lower),
        )
        if violation >= material:
            violations.append(int(gen_id))
            maximum = max(maximum, violation)
    expected = label.q_unlimited_counterfactual
    structural = evaluate_solved_feasibility(solved)
    if (
        tuple(violations) != expected.material_violation_gen_ids
        or not _close(maximum, expected.max_violation_mvar, tolerance=1e-6)
        or not _ext_grid_feasible(solved)
        or not structural.generator_p_within_limits
        or not structural.connected
        or not structural.loads_energized
    ):
        _fail(report, 9, "Q-unlimited material evidence differs")


def _verify_qv_evidence(
    label: ScenarioLabel,
    base_net: Any,
    evidence_net: Any,
    raw_policy: dict[str, Any],
    report: ScenarioValidation,
) -> None:
    base = evaluate_solved_feasibility(base_net)
    evidence = evaluate_solved_feasibility(evidence_net)
    expected = label.qv_evidence
    threshold = raw_policy["reactive_thresholds"]
    voltages = evidence_net.res_bus["vm_pu"].dropna().astype(float)
    minimum = float(voltages.min())
    weak_ids = tuple(
        int(bus_id)
        for bus_id in sorted(
            voltages.index[
                voltages
                <= minimum + float(threshold["weak_bus_band_pu"])
            ]
        )
    )
    base_by_id = {item.gen_id: item for item in base.generator_q_status}
    evidence_by_id = {
        item.gen_id: item
        for item in evidence.generator_q_status
    }
    common = sorted(set(base_by_id) & set(evidence_by_id))
    reduction = math.fsum(
        max(float(base_by_id[gen_id].upper_headroom_mvar or 0.0), 0.0)
        - max(
            float(
                evidence_by_id[gen_id].upper_headroom_mvar or 0.0
            ),
            0.0,
        )
        for gen_id in common
    )
    limited = tuple(evidence.q_limited_gen_ids)
    newly = tuple(
        gen_id
        for gen_id in limited
        if gen_id not in base.q_limited_gen_ids
    )
    upper_near = tuple(
        item.gen_id
        for item in evidence.generator_q_status
        if item.status == "Q_LIMITED_UPPER"
        or (
            item.status == "PV_CONTROLLABLE"
            and item.upper_headroom_mvar is not None
            and item.upper_headroom_mvar
            < max(
                float(threshold["material_q_violation_floor_mvar"]),
                float(threshold["material_q_violation_fraction"])
                * abs(item.max_q_mvar - item.min_q_mvar),
            )
        )
    )
    distance = impedance_weighted_graph_distances(
        evidence_net,
        source_buses=(label.recipe.pocket.anchor_bus,),
        policy=ElectricalDistancePolicy(
            common_mva_base=float(threshold["common_mva_base"]),
            minimum_edge_weight_pu=float(
                threshold["minimum_edge_weight_pu"]
            ),
        ),
    )
    weak_distances = tuple(
        float(observed_distance)
        for bus_id in weak_ids
        if (
            observed_distance := distance.distances_pu.get(bus_id)
        )
        is not None
    )
    minimum_weak_distance = (
        min(weak_distances) if weak_distances else math.inf
    )
    local = minimum_weak_distance <= float(
        threshold["maximum_weak_region_distance_pu"]
    )
    if (
        weak_ids != expected.weak_bus_ids
        or not _close(minimum, expected.min_vm_pu, tolerance=1e-8)
        or not _close(
            reduction,
            expected.q_headroom_reduction_mvar,
            tolerance=1e-6,
        )
        or upper_near != expected.q_near_limit_gen_ids
        or limited != expected.q_limited_gen_ids
        or newly != expected.newly_q_limited_gen_ids
        or not _close(
            minimum_weak_distance,
            expected.weak_region_min_distance_pu,
        )
        or local is not expected.weak_region_local
        or base.voltage.min_vm_pu - minimum
        < float(threshold["minimum_voltage_deterioration_pu"])
    ):
        _fail(report, 10, "constrained Q-V evidence differs")
    slack = evidence.slack_results[0]
    if not (
        _close(slack.p_mw, expected.ext_grid_p_mw, tolerance=1e-6)
        and _close(slack.q_mvar, expected.ext_grid_q_mvar, tolerance=1e-6)
    ):
        _fail(report, 10, "Q-V external-grid evidence differs")


def _verify_witness(
    label: ScenarioLabel,
    raw_witness: dict[str, Any],
    target_net: Any,
    raw_policy: dict[str, Any],
    report: ScenarioValidation,
) -> None:
    maneuvers = raw_witness.get("maneuvers", [])
    state_hashes = raw_witness.get("state_hashes", [])
    if (
        len(maneuvers) != label.witness_length
        or len(state_hashes) != len(maneuvers) + 1
        or state_hashes[0] != state_fingerprint(target_net).value
    ):
        _fail(report, 12, "witness lengths or initial hash differ")
        return
    current = target_net
    for index, raw_action in enumerate(maneuvers, start=1):
        q_context = _witness_action_q_context(
            current,
            raw_policy,
        )
        legal = enumerate_legal_qv_actions(current, q_context)
        action = next(
            (
                candidate
                for candidate in legal
                if candidate.model_dump(mode="json") == raw_action
            ),
            None,
        )
        if action is None:
            _fail(report, 12, f"witness action {index} is not legal")
            return
        applicability = get_qv_action_applicability(
            current,
            action,
            q_context,
        )
        if not applicability.applicable:
            _fail(report, 12, f"witness action {index} is inapplicable")
            return
        current = apply_qv_action(current, action, q_context)
        if state_fingerprint(current).value != state_hashes[index]:
            _fail(report, 12, f"witness state hash {index} differs")
    terminal = solve_locked_probe(current)
    if terminal.status != "SOLVED":
        _fail(report, 12, "witness terminal does not converge")
    else:
        terminal_feasibility = evaluate_solved_feasibility(
            terminal.solved_net
        )
        if not satisfies_non_voltage_constraints(terminal_feasibility):
            _fail(
                report,
                12,
                "witness terminal violates a non-voltage feasibility constraint",
            )
        else:
            _verify_terminal_pf(terminal, raw_witness, report)
    direct = _has_direct_restorer(
        target_net,
        raw_policy["witness"],
    )
    _verify_witness_optimality(
        label,
        direct_restorer_available=direct,
        report=report,
    )
    if (
        direct != label.direct_restorer_available
        or (direct and label.resolution_regime != "DIRECT")
        or (not direct and label.resolution_regime != "SEQUENTIAL")
    ):
        _fail(report, 12, "resolution regime differs from direct audit")
    offset = label.recipe.target_stress - label.convergence_boundary.upper
    if not (
        _close(offset, label.target_depth.stress_offset)
        and _close(
            offset / label.convergence_boundary.upper,
            label.target_depth.relative_offset,
        )
    ):
        _fail(report, 12, "target depth differs")


def _verify_terminal_pf(
    terminal: Any,
    raw_witness: dict[str, Any],
    report: ScenarioValidation,
) -> None:
    """Re-solve the witness terminal and confront the stored verdict with the observed one."""
    stored = raw_witness.get("terminal_pf")
    if not isinstance(stored, dict):
        _fail(report, 12, "witness terminal_pf is missing")
        return
    if bool(stored.get("converged")) is not True:
        _fail(report, 12, "stored witness terminal_pf does not record convergence")
        return
    stored_feasibility = stored.get("feasibility")
    stored_voltage = (
        stored_feasibility.get("voltage")
        if isinstance(stored_feasibility, dict)
        else None
    )
    if not isinstance(stored_voltage, dict):
        _fail(
            report,
            12,
            "stored witness terminal_pf is missing voltage metrics",
        )
        return

    observed = evaluate_solved_feasibility(terminal.solved_net).voltage
    stored_min = stored_voltage.get("min_vm_pu")
    stored_max = stored_voltage.get("max_vm_pu")
    stored_hard_ok = stored_voltage.get("hard_envelope_ok")
    if (
        not _close(stored_min, observed.min_vm_pu)
        or not _close(stored_max, observed.max_vm_pu)
        or stored_hard_ok is not observed.hard_envelope_ok
    ):
        _fail(
            report,
            12,
            "stored witness terminal voltage metrics differ from independent solve",
        )
    report.details.setdefault(
        "witness_terminal_voltage",
        [],
    ).append(
        f"min_vm_pu={observed.min_vm_pu:.12g}, "
        f"max_vm_pu={observed.max_vm_pu:.12g}, "
        f"hard_envelope_ok={observed.hard_envelope_ok}"
    )


def _verify_witness_optimality(
    label: ScenarioLabel,
    *,
    direct_restorer_available: bool,
    report: ScenarioValidation,
) -> None:
    """Certify each optimality claim the artifacts make verifiable, and reject the rest.

    Minimality at length 1 and 2 is decidable from the direct-restorer audit this validator
    already performs: a length-1 witness is minimal, and a length-2 witness is minimal exactly
    when no single action restores. Beyond that, certifying EXACT_MINIMUM needs an exhaustive
    search over the shallower depths, which the artifacts do not carry.
    """
    if label.witness_optimality == "UPPER_BOUND":
        if label.resolution_regime == "DIRECT":
            _fail(report, 12, "DIRECT witness cannot be an UPPER_BOUND")
        return

    if label.resolution_regime == "DIRECT":
        if label.witness_length != 1:
            _fail(report, 12, "DIRECT witness claims EXACT_MINIMUM with length above one")
        return

    if label.witness_length == 2:
        if direct_restorer_available:
            _fail(report, 12, "length-2 EXACT_MINIMUM claim contradicts an available direct restorer")
        return

    _fail(
        report,
        12,
        f"EXACT_MINIMUM at witness length {label.witness_length} is not certifiable from the "
        "recorded artifacts; record it as UPPER_BOUND or carry exhaustive-search evidence",
    )


def _diagnostic_q_context(
    snapshot: Any,
    policy: dict[str, Any],
) -> dict[int, str]:
    active = _active_policy_from_net(snapshot)

    def builder(value: float) -> Any:
        return build_diagnostic_state(
            snapshot,
            lambda_value=value,
            active_policy=active,
        )

    measured = measure_boundary(
        builder,
        coarse_coordinates=tuple(policy["diagnostic_coordinates"]),
        refinement_resolution=float(policy["diagnostic_resolution"]),
        feasibility_policy=BoundaryFeasibilityPolicy(),
    )
    if measured.status != "BOUNDARY_FOUND" or measured.highest_solved is None:
        return {}
    evidence = builder(measured.highest_solved)
    probe = solve_locked_probe(evidence.net)
    if probe.status != "SOLVED":
        return {}
    feasibility = evaluate_solved_feasibility(probe.solved_net)
    if not feasibility.feasible:
        return {}
    return {
        item.gen_id: item.status
        for item in feasibility.generator_q_status
    }


def _witness_action_q_context(
    snapshot: Any,
    raw_policy: dict[str, Any],
) -> dict[int, str]:
    """Use direct Q evidence for solved states and retreat evidence otherwise."""
    probe = solve_locked_probe(snapshot)
    if probe.status == "SOLVED":
        return {
            item.gen_id: item.status
            for item in evaluate_solved_feasibility(
                probe.solved_net
            ).generator_q_status
        }
    return _diagnostic_q_context(snapshot, raw_policy["witness"])


def _active_policy_from_net(net: Any) -> ActiveBalancePolicy:
    participants = []
    for gen_id, row in net.gen.sort_index().iterrows():
        if (
            bool(row.get("in_service", True))
            and float(row["max_p_mw"]) > float(row["min_p_mw"])
            and float(row["max_p_mw"]) > 0.0
        ):
            if float(row["p_mw"]) <= 0.0:
                raise ValueError("diagnostic participant has non-positive P")
            participants.append(
                GeneratorParticipation(
                    gen_id=int(gen_id),
                    factor=float(row["p_mw"]),
                )
            )
    return ActiveBalancePolicy(participation=tuple(participants))


def _has_direct_restorer(
    target: Any,
    witness_policy: dict[str, Any],
) -> bool:
    q_context = _diagnostic_q_context(target, witness_policy)
    for action in enumerate_legal_qv_actions(target, q_context):
        changed = apply_qv_action(target, action, q_context)
        probe = solve_locked_probe(changed)
        if probe.status == "SOLVED" and satisfies_non_voltage_constraints(
            evaluate_solved_feasibility(probe.solved_net)
        ):
            return True
    return False


def _verify_artifacts(
    entry: Any,
    full: Any,
    full_path: Path,
    lean_path: Path,
    card_path: Path,
    report: ScenarioValidation,
) -> None:
    if (
        _sha256_file(full_path) != entry.full_artifact_hash
        or _sha256_file(lean_path) != entry.lean_artifact_hash
        or _sha256_file(card_path) != entry.card_artifact_hash
    ):
        _fail(report, 13, "public artifact hash differs")
    lean = pp.from_json(str(lean_path))
    for table, columns in LEAN_COLUMNS.items():
        if set(getattr(lean, table).columns) != columns:
            _fail(report, 13, f"LEAN {table} columns differ")
            continue
        lean_table = getattr(lean, table)
        full_table = getattr(full, table).loc[
            lean_table.index,
            lean_table.columns,
        ]
        if not lean_table.equals(full_table):
            _fail(report, 13, f"LEAN {table} values differ from FULL")
    if not lean.poly_cost.empty:
        _fail(report, 13, "LEAN poly_cost is not empty")
    if card_path.read_text(encoding="utf-8") != render_scenario_card(full):
        _fail(report, 13, "Scenario Card differs from independent render")


def _verify_private_denial(
    full_path: Path,
    lean_path: Path,
    card_path: Path,
    report: ScenarioValidation,
) -> None:
    text = "\n".join(
        (
            full_path.read_text(encoding="utf-8"),
            lean_path.read_text(encoding="utf-8"),
            card_path.read_text(encoding="utf-8"),
        )
    ).lower()
    hits = [token for token in PRIVATE_TOKENS if token in text]
    if hits:
        _fail(report, 14, f"public artifacts expose private tokens: {hits}")


def _check_shape_and_results(
    net: Any,
    report: ScenarioValidation,
    *,
    network_id: str = "case118",
) -> None:
    expected_by_network = {
        "case118": {"bus": 118, "load": 99, "gen": 53, "ext_grid": 1, "shunt": 14, "trafo": 13},
        "case89pegase": {"bus": 89, "load": 29, "gen": 11, "ext_grid": 1, "shunt": 44, "trafo": 50},
    }
    try:
        expected = expected_by_network[network_id]
    except KeyError as exc:
        raise ValueError(f"unsupported dataset network: {network_id}") from exc
    for table, count in expected.items():
        if len(getattr(net, table)) != count:
            _fail(report, 1, f"{table} count differs from {count}")
    for name in sorted(key for key in net if key.startswith("res_")):
        table = net[name]
        if hasattr(table, "empty") and not table.empty:
            _fail(report, 13, f"FULL persists result table {name}")


def _global_contract_failures(
    *,
    root: Path,
    manifest: DatasetManifest,
    evaluation: EvaluationManifest,
    labels: list[ScenarioLabel],
    witnesses: list[dict[str, Any]],
    raw_policy: dict[str, Any],
    declared_split_counts: dict[str, int],
) -> dict[str, list[str]]:
    ids = [entry.scenario_id for entry in manifest.scenarios]
    failures: dict[str, list[str]] = defaultdict(list)
    label_ids = [label.scenario_id for label in labels]
    witness_ids = [str(row.get("scenario_id")) for row in witnesses]
    evaluation_ids = [entry.scenario_id for entry in evaluation.scenarios]
    if not (ids == label_ids == witness_ids == evaluation_ids):
        for scenario_id in ids:
            failures[scenario_id].append(
                "manifest, split, labels, and witnesses IDs differ"
            )
    if _sha256_file(root / "evaluation_manifest.json") != manifest.split_manifest_hash:
        for scenario_id in ids:
            failures[scenario_id].append("split manifest hash differs")
    # The runtime reads the split from the evaluation manifest while curation records it on the
    # private label. A self-consistent but disagreeing pair would pass every hash check and still
    # route scenarios into the wrong split, so compare the two sources directly.
    if manifest.dataset_version != evaluation.dataset_version:
        for scenario_id in ids:
            failures[scenario_id].append("public and evaluation manifest dataset versions differ")
    label_splits = {label.scenario_id: label.memory_split for label in labels}
    for entry in evaluation.scenarios:
        expected_split = label_splits.get(entry.scenario_id)
        if expected_split is not None and entry.memory_split != expected_split:
            failures[entry.scenario_id].append(
                "evaluation manifest memory_split differs from the curation label"
            )
    expected_policy_hashes = {
        "generation": _sha256_payload(raw_policy),
        "operating_profiles": _sha256_payload(
            raw_policy["operating_profiles"]
        ),
        "pockets": _sha256_payload(raw_policy["pockets"]),
        "scan": _sha256_payload(raw_policy["scan"]),
        "reactive_thresholds": _sha256_payload(
            raw_policy["reactive_thresholds"]
        ),
        "witness": _sha256_payload(raw_policy["witness"]),
        "composition": _sha256_payload(raw_policy["composition"]),
    }
    if manifest.policy_hashes != expected_policy_hashes:
        for scenario_id in ids:
            failures[scenario_id].append("generation policy hashes differ")
    if labels:
        augmented = _independent_augmented_base(str(raw_policy.get("network_id", "case118")))
        augmentation_version = (
            labels[0]
            .generation_metadata.curation_policy_versions.augmentation
        )
        base_hash = state_fingerprint(
            augmented,
            policy_versions={"augmentation": augmentation_version},
        ).value
        if manifest.base_network_hash != base_hash:
            for scenario_id in ids:
                failures[scenario_id].append(
                    "augmented base-network hash differs"
                )

    groups = _independent_leakage_groups(
        labels,
        float(raw_policy["leakage_similarity_threshold"]),
    )
    split_by_group: dict[str, set[str]] = defaultdict(set)
    for label in labels:
        expected_group = groups[label.scenario_id]
        if label.leakage_group_id != expected_group:
            failures[label.scenario_id].append("leakage group differs")
        split_by_group[expected_group].add(label.memory_split)
    broken_groups = {
        group
        for group, splits in split_by_group.items()
        if len(splits) != 1
    }
    for label in labels:
        if groups[label.scenario_id] in broken_groups:
            failures[label.scenario_id].append("leakage group crosses split")
    if not _split_counts_match(labels, declared_split_counts):
        for scenario_id in ids:
            failures[scenario_id].append("split counts differ")
    return failures


def _split_counts_match(
    labels: Sequence[Any],
    declared: dict[str, int],
) -> bool:
    observed = Counter(str(label.memory_split) for label in labels)
    return all(observed[split] == count for split, count in declared.items())


def _independent_leakage_groups(
    labels: list[ScenarioLabel],
    threshold: float,
) -> dict[str, str]:
    parent = list(range(len(labels)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(
                left_root,
                right_root,
            )

    for left_index, left in enumerate(labels):
        for right_index in range(left_index + 1, len(labels)):
            right = labels[right_index]
            if (
                left.scenario_family_id == right.scenario_family_id
                or _pocket_similarity(
                    left.recipe.pocket,
                    right.recipe.pocket,
                )
                >= threshold
            ):
                union(left_index, right_index)
    families: dict[int, set[str]] = defaultdict(set)
    for index, label in enumerate(labels):
        families[find(index)].add(label.scenario_family_id)
    group_ids = {
        root: (
            "L-"
            + hashlib.sha256(
                "\n".join(sorted(values)).encode()
            ).hexdigest()[:20]
        )
        for root, values in families.items()
    }
    return {
        label.scenario_id: group_ids[find(index)]
        for index, label in enumerate(labels)
    }


def _pocket_similarity(left: Any, right: Any) -> float:
    left_by_id = {point.load_id: point.weight for point in left.loads}
    right_by_id = {point.load_id: point.weight for point in right.loads}
    ids = sorted(set(left_by_id) | set(right_by_id))
    dot = math.fsum(
        left_by_id.get(load_id, 0.0)
        * right_by_id.get(load_id, 0.0)
        for load_id in ids
    )
    left_norm = math.sqrt(
        math.fsum(left_by_id.get(load_id, 0.0) ** 2 for load_id in ids)
    )
    right_norm = math.sqrt(
        math.fsum(right_by_id.get(load_id, 0.0) ** 2 for load_id in ids)
    )
    return dot / (left_norm * right_norm)


def _ext_grid_feasible(net: Any) -> bool:
    for ext_grid_id, row in net.ext_grid.sort_index().iterrows():
        if not bool(row.get("in_service", True)):
            continue
        for value_name, lower_name, upper_name in (
            ("p_mw", "min_p_mw", "max_p_mw"),
            ("q_mvar", "min_q_mvar", "max_q_mvar"),
        ):
            value = float(net.res_ext_grid.at[ext_grid_id, value_name])
            lower = float(row.get(lower_name, float("nan")))
            upper = float(row.get(upper_name, float("nan")))
            if (
                not math.isfinite(value)
                or (math.isfinite(lower) and value < lower - 1e-3)
                or (math.isfinite(upper) and value > upper + 1e-3)
            ):
                return False
    return True


def _load_generation_report(root: Path) -> dict[str, Any]:
    report = json.loads(
        (root / "generation_report.json").read_text(encoding="utf-8")
    )
    if not isinstance(report, dict):
        raise ValueError("generation_report must contain an object")
    return report


def _generation_policy(report: dict[str, Any]) -> dict[str, Any]:
    policy = report.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("generation_report policy is missing")
    return policy


def _declared_split_counts(
    report: dict[str, Any],
    *,
    scenario_count: int,
) -> dict[str, int]:
    stored = report.get("split_counts")
    if stored is None:
        held_out = scenario_count // 4
        return {
            "memory_population": scenario_count - held_out,
            "held_out": held_out,
        }
    if not isinstance(stored, dict):
        raise ValueError("generation_report split_counts must contain an object")
    expected_keys = {"memory_population", "held_out"}
    if set(stored) != expected_keys:
        raise ValueError("generation_report split_counts has unexpected keys")
    counts = {key: stored[key] for key in sorted(expected_keys)}
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values()):
        raise ValueError("generation_report split_counts must be non-negative integers")
    if sum(counts.values()) != scenario_count:
        raise ValueError("generation_report split_counts must sum to scenario_count")
    return counts


def _load_models(path: Path, model_type: Any) -> list[Any]:
    values = _load_raw_list(path)
    return [model_type.model_validate(value) for value in values]


def _load_raw_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(
        isinstance(item, dict)
        for item in value
    ):
        raise ValueError(f"{path} must contain a list of objects")
    return value


def _fail(
    report: ScenarioValidation,
    check_number: int,
    message: str,
) -> None:
    check = f"check_{check_number}_{CHECKS[check_number]}"
    if check not in report.failed_checks:
        report.failed_checks.append(check)
    report.details.setdefault(check, []).append(message)


def _finish(report: ScenarioValidation) -> ScenarioValidation:
    report.valid = not report.failed_checks
    return report


def _close(
    left: Any,
    right: Any,
    *,
    tolerance: float = 1e-9,
) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(
        float(left),
        float(right),
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def write_validation_report(
    report: CorpusValidation,
    dataset_dir: Path,
) -> Path:
    path = dataset_dir / "validation_report.json"
    atomic_write_json(
        path,
        report.model_dump(mode="json"),
    )
    return path


def _validation_checkpoint_identity(root: Path) -> dict[str, Any]:
    required = (
        "manifest.json",
        "evaluation_manifest.json",
        "generation_report.json",
        "private/labels.json",
        "private/witnesses.json",
    )
    return {
        "kind": "reactive-deficit-validation",
        "validator_version": VALIDATOR_VERSION,
        "artifact_hashes": {
            relative: _sha256_file(root / relative)
            for relative in required
        },
    }


def _require_safe_validation_checkpoint(checkpoint_dir: Path) -> None:
    resolved = checkpoint_dir.resolve()
    frozen = (
        Path(__file__).resolve().parents[2] / "dataset/ieee118"
    ).resolve()
    if resolved == frozen or frozen in resolved.parents:
        raise ValueError(
            "refusing to place validation checkpoints in the frozen corpus"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Explicit staged dataset directory.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Durable per-scenario validation checkpoint directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an identity-compatible validation checkpoint.",
    )
    args = parser.parse_args(argv)
    if args.resume and args.checkpoint_dir is None:
        parser.error("--resume requires --checkpoint-dir")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        report = validate_corpus(
            args.dataset_dir,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
        )
    except (FileNotFoundError, ValidationError, ValueError) as exc:
        print(f"Validation could not start: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    write_validation_report(report, args.dataset_dir)
    print(f"{report.valid_count} valid / {report.invalid_count} invalid")
    raise SystemExit(0 if report.valid else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
