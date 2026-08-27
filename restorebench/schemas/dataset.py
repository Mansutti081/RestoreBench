# ABOUTME: Defines private reactive-deficit curation records and public dataset manifests.
# ABOUTME: Keeps private recipes and witnesses structurally absent from runtime Scenario objects.
from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from restorebench.schemas.actions import Action
from restorebench.schemas.power_flow import PowerFlowResult


ScenarioId = Annotated[str, Field(pattern=r"^S\d{4}$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
MemorySplit = Literal["memory_population", "held_out"]


class SolverSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm: Literal["nr"] = "nr"
    enforce_q_lims: Literal[True] = True
    init: Literal["dc"] = "dc"
    max_iteration: Literal[30] = 30
    primary_tolerance_mva: Literal[1e-8] = 1e-8
    recovery_tolerance_mva: Literal[1e-6] = 1e-6
    check_connectivity: Literal[True] = True


class SharedPhysicsPolicyVersions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    active_balance: Annotated[str, Field(min_length=1)]
    action: Annotated[str, Field(min_length=1)]
    solver_probe: Annotated[str, Field(min_length=1)]
    feasibility: Annotated[str, Field(min_length=1)]
    electrical_distance: Annotated[str, Field(min_length=1)]
    fingerprint: Annotated[str, Field(min_length=1)]


class CurationPolicyVersions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    augmentation: Annotated[str, Field(min_length=1)]
    operating_profile: Annotated[str, Field(min_length=1)]
    pocket_weighting: Annotated[str, Field(min_length=1)]
    load_stress: Annotated[str, Field(min_length=1)]
    alternative_init: Annotated[str, Field(min_length=1)]
    monotonicity_scan: Annotated[str, Field(min_length=1)]
    qv_thresholds: Annotated[str, Field(min_length=1)]
    witness_search: Annotated[str, Field(min_length=1)]
    composition: Annotated[str, Field(min_length=1)]
    split: Annotated[str, Field(min_length=1)]


class CurationLoadPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    load_id: int
    base_p_mw: float = Field(allow_inf_nan=False)
    base_q_mvar: float = Field(allow_inf_nan=False)
    weight: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class BaseGeneratorDispatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gen_id: int
    base_p_mw: float = Field(allow_inf_nan=False)


class PocketRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    anchor_bus: int
    distance_method: Literal["IMPEDANCE_WEIGHTED_GRAPH_DISTANCE"]
    loads: tuple[CurationLoadPoint, ...]
    vector_hash: Sha256
    policy_version: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def loads_are_complete_and_ordered(self) -> "PocketRecipe":
        ids = [point.load_id for point in self.loads]
        if not ids:
            raise ValueError("pocket must contain at least one load")
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("pocket load IDs must be unique and in ascending order")
        if max(point.weight for point in self.loads) != 1.0:
            raise ValueError("pocket weights must be normalized so max(weight)=1")
        if not any(point.weight > 0.0 for point in self.loads):
            raise ValueError("pocket must contain non-zero load support")
        return self


class ActiveScheduleRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    generators: tuple[BaseGeneratorDispatch, ...]
    participation_factors: tuple[Annotated[float, Field(gt=0.0, allow_inf_nan=False)], ...]
    policy_version: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def generators_match_participation(self) -> "ActiveScheduleRecipe":
        ids = [generator.gen_id for generator in self.generators]
        if not ids:
            raise ValueError("active schedule requires at least one participating generator")
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("active-schedule generator IDs must be unique and in ascending order")
        if len(self.generators) != len(self.participation_factors):
            raise ValueError("participation factors must match participating generators one-for-one")
        return self


class CurationRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operating_profile_id: Annotated[str, Field(min_length=1)]
    base_state_hash: Sha256
    pocket: PocketRecipe
    active_schedule: ActiveScheduleRecipe
    target_stress: float = Field(gt=0.0, allow_inf_nan=False)
    recipe_hash: Sha256


class BoundaryInterval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lower: float = Field(ge=0.0, allow_inf_nan=False)
    upper: float = Field(gt=0.0, allow_inf_nan=False)
    resolution: float = Field(gt=0.0, allow_inf_nan=False)
    capped: bool

    @model_validator(mode="after")
    def interval_is_ordered(self) -> "BoundaryInterval":
        if self.upper <= self.lower:
            raise ValueError("boundary upper must be greater than lower")
        if self.upper - self.lower > self.resolution + 1e-12:
            raise ValueError("boundary width must not exceed its declared resolution")
        return self


class MonotonicityObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal[
        "OBSERVED_MONOTONIC",
        "OBSERVED_NON_MONOTONIC",
        "INSUFFICIENT",
    ]
    probe_coordinates: tuple[float, ...]
    probe_statuses: tuple[Literal["SOLVED", "NO_SOLUTION", "INFEASIBLE"], ...]
    policy_version: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def probes_are_aligned_and_ordered(self) -> "MonotonicityObservation":
        if len(self.probe_coordinates) != len(self.probe_statuses):
            raise ValueError("probe coordinates and statuses must have equal lengths")
        if not self.probe_coordinates:
            raise ValueError("monotonicity observation requires at least one probe")
        if not all(math.isfinite(value) for value in self.probe_coordinates):
            raise ValueError("probe coordinates must be finite")
        if any(
            right <= left
            for left, right in zip(
                self.probe_coordinates,
                self.probe_coordinates[1:],
                strict=False,
            )
        ):
            raise ValueError("probe coordinates must be strictly increasing")
        observed_non_monotonic = _has_solved_unsolved_solved(self.probe_statuses)
        if observed_non_monotonic != (self.status == "OBSERVED_NON_MONOTONIC"):
            raise ValueError("monotonicity status must match the ordered probe record")
        if self.status == "OBSERVED_MONOTONIC" and len(self.probe_coordinates) < 2:
            raise ValueError("observed monotonicity requires at least two probes")
        return self


class QVEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_stress: float = Field(ge=0.0, allow_inf_nan=False)
    weak_bus_ids: tuple[int, ...]
    weak_region_min_distance_pu: float = Field(
        ge=0.0,
        allow_inf_nan=False,
    )
    weak_region_local: bool
    min_vm_pu: float = Field(ge=0.0, allow_inf_nan=False)
    q_near_limit_gen_ids: tuple[int, ...]
    q_limited_gen_ids: tuple[int, ...]
    newly_q_limited_gen_ids: tuple[int, ...]
    q_headroom_reduction_mvar: float = Field(gt=0.0, allow_inf_nan=False)
    generator_p_feasible: Literal[True]
    ext_grid_p_mw: float = Field(allow_inf_nan=False)
    ext_grid_q_mvar: float = Field(allow_inf_nan=False)
    hard_voltage_envelope_passed: Literal[True]
    thresholds_version: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def evidence_ids_are_stable(self) -> "QVEvidence":
        for name in (
            "weak_bus_ids",
            "q_near_limit_gen_ids",
            "q_limited_gen_ids",
            "newly_q_limited_gen_ids",
        ):
            values = getattr(self, name)
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must contain unique IDs in ascending order")
        if not self.weak_bus_ids:
            raise ValueError("Q-V evidence requires at least one weak bus")
        if not (self.q_near_limit_gen_ids or self.q_limited_gen_ids):
            raise ValueError("Q-V evidence requires a near-limited or Q-limited generator")
        if not set(self.newly_q_limited_gen_ids) <= set(self.q_limited_gen_ids):
            raise ValueError("newly Q-limited generators must be Q-limited")
        return self


class QUnlimitedCounterfactual(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    converged: Literal[True]
    material_violation_gen_ids: tuple[int, ...]
    max_violation_mvar: float = Field(gt=0.0, allow_inf_nan=False)
    ext_grid_feasible: Literal[True]
    policy_version: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def material_violations_are_present(self) -> "QUnlimitedCounterfactual":
        if not self.material_violation_gen_ids:
            raise ValueError("Q-unlimited counterfactual requires a material generator-Q violation")
        if tuple(sorted(set(self.material_violation_gen_ids))) != self.material_violation_gen_ids:
            raise ValueError("material violation generator IDs must be unique and ascending")
        return self


class AlternativeInitializationAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    init_policy: Annotated[str, Field(min_length=1)]
    primary_status: Literal["SOLVED", "NO_SOLUTION"]
    recovery_status: Literal["SOLVED", "NO_SOLUTION", "NOT_RUN"]
    converged_without_action: bool
    solver_attempt_count: int = Field(ge=1, le=2)

    @model_validator(mode="after")
    def attempt_record_is_consistent(self) -> "AlternativeInitializationAudit":
        converged = self.primary_status == "SOLVED" or self.recovery_status == "SOLVED"
        if converged != self.converged_without_action:
            raise ValueError("alternative-init convergence flag must match attempt statuses")
        if self.primary_status == "SOLVED":
            if self.recovery_status != "NOT_RUN" or self.solver_attempt_count != 1:
                raise ValueError("recovery must not run after alternative-init primary success")
        elif self.recovery_status == "NOT_RUN" or self.solver_attempt_count != 2:
            raise ValueError("alternative-init primary failure requires one recovery attempt")
        return self


class TargetDepth(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stress_offset: float = Field(gt=0.0, allow_inf_nan=False)
    relative_offset: float = Field(gt=0.0, allow_inf_nan=False)
    policy_version: Annotated[str, Field(min_length=1)]


class GenerationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    generator_version: Annotated[str, Field(min_length=1)]
    validator_version: Annotated[str, Field(min_length=1)]
    python_version: Annotated[str, Field(min_length=1)]
    pandapower_version: Annotated[str, Field(min_length=1)]
    seed: Literal[42]
    solver_settings: SolverSettings
    shared_policy_versions: SharedPhysicsPolicyVersions
    curation_policy_versions: CurationPolicyVersions


class ScenarioLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: ScenarioId
    scenario_class: Literal["REACTIVE_DEFICIT"]
    scenario_family_id: Annotated[str, Field(min_length=1)]
    leakage_group_id: Annotated[str, Field(min_length=1)]
    recipe: CurationRecipe
    convergence_boundary: BoundaryInterval
    monotonicity: MonotonicityObservation
    qv_evidence: QVEvidence
    q_unlimited_counterfactual: QUnlimitedCounterfactual
    alternative_init_audit: AlternativeInitializationAudit
    resolvable_within_budget: Literal[True]
    resolution_regime: Literal["DIRECT", "SEQUENTIAL"]
    direct_restorer_available: bool
    witness_length: int = Field(ge=1, le=10)
    witness_optimality: Literal["EXACT_MINIMUM", "UPPER_BOUND"]
    target_depth: TargetDepth
    memory_split: MemorySplit
    generation_metadata: GenerationMetadata

    @model_validator(mode="after")
    def resolution_and_depth_are_consistent(self) -> "ScenarioLabel":
        if self.resolution_regime == "DIRECT":
            if (
                not self.direct_restorer_available
                or self.witness_length != 1
                or self.witness_optimality != "EXACT_MINIMUM"
            ):
                raise ValueError(
                    "DIRECT requires an available direct restorer, witness length 1, "
                    "and EXACT_MINIMUM optimality"
                )
        elif self.direct_restorer_available or self.witness_length < 2:
            raise ValueError(
                "SEQUENTIAL requires no direct restorer and witness length between 2 and 10"
            )

        expected_offset = self.recipe.target_stress - self.convergence_boundary.upper
        expected_relative = expected_offset / self.convergence_boundary.upper
        if expected_offset <= 0.0 or not math.isclose(
            self.target_depth.stress_offset,
            expected_offset,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("target depth must match target stress minus boundary upper")
        if not math.isclose(
            self.target_depth.relative_offset,
            expected_relative,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("target depth relative offset must match the boundary")
        if self.qv_evidence.evidence_stress > self.convergence_boundary.lower + 1e-9:
            raise ValueError("Q-V evidence stress must not exceed the convergent boundary")
        if self.monotonicity.status != "OBSERVED_MONOTONIC":
            raise ValueError("admitted scenarios require observed monotonicity")
        if self.alternative_init_audit.converged_without_action:
            raise ValueError("admitted scenarios must fail the alternative-initialization audit")
        return self


class CurationWitness(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: ScenarioId
    maneuvers: tuple[Action, ...] = Field(min_length=1, max_length=10)
    state_hashes: tuple[Sha256, ...]
    terminal_pf: PowerFlowResult
    search_policy_version: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def witness_record_is_consistent(self) -> "CurationWitness":
        if len(self.state_hashes) != len(self.maneuvers) + 1:
            raise ValueError("state_hashes must contain the initial state and one hash per maneuver")
        if not self.terminal_pf.converged:
            raise ValueError("curation witness terminal power flow must converge")
        return self


class PublicScenarioEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: ScenarioId
    full_artifact_hash: Sha256
    lean_artifact_hash: Sha256
    card_artifact_hash: Sha256


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_version: Annotated[str, Field(min_length=1)]
    base_network_hash: Sha256
    scenario_count: int = Field(ge=1)
    scenarios: tuple[PublicScenarioEntry, ...]
    split_manifest_hash: Sha256
    policy_hashes: dict[str, Sha256]
    environment: dict[str, str]

    @model_validator(mode="after")
    def scenarios_match_declared_count(self) -> "DatasetManifest":
        ids = [scenario.scenario_id for scenario in self.scenarios]
        if self.scenario_count != len(ids):
            raise ValueError("scenario_count must equal the number of scenario entries")
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("manifest scenario IDs must be unique and in ascending order")
        return self


class EvaluationScenarioEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: ScenarioId
    memory_split: MemorySplit


class EvaluationManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_version: Annotated[str, Field(min_length=1)]
    scenarios: tuple[EvaluationScenarioEntry, ...]

    @model_validator(mode="after")
    def scenario_ids_are_unique_and_ordered(self) -> "EvaluationManifest":
        ids = [scenario.scenario_id for scenario in self.scenarios]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("evaluation scenario IDs must be unique and in ascending order")
        return self


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: ScenarioId
    dataset_version: Annotated[str, Field(min_length=1)] | None = None
    full_net_path: str
    card_path: str
    memory_split: MemorySplit


def _has_solved_unsolved_solved(
    statuses: tuple[Literal["SOLVED", "NO_SOLUTION", "INFEASIBLE"], ...],
) -> bool:
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
